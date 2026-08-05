import multiprocessing
import os
from time import sleep
from typing import List, Type, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile, maybe_mkdir_p
from deep_utils import DirUtils, StringUtils
from tqdm import tqdm

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets


class DatasetFingerprintExtractor:
    def __init__(self, dataset_name_or_id: Union[str, int], num_processes: int = 8, verbose: bool = False, num_foreground_voxels_for_intensitystats=10e7):
        """
        extracts the dataset fingerprint used for experiment planning. The dataset fingerprint will be saved as a
        json file in the input_folder

        Philosophy here is to do only what we really need. Don't store stuff that we can easily read from somewhere
        else. Don't compute stuff we don't need (except for intensity_statistics_per_channel)
        """
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
        self.verbose = verbose

        self.dataset_name = dataset_name
        self.input_folder = join(nnUNet_raw, dataset_name)
        self.num_processes = num_processes
        self.dataset_json = load_json(join(self.input_folder, 'dataset.json'))
        self.dataset = get_filenames_of_train_images_and_targets(self.input_folder, self.dataset_json)

        # We don't want to use all foreground voxels because that can accumulate a lot of data (out of memory). It is
        # also not critically important to get all pixels as long as there are enough. Let's use 10e7 voxels in total
        # (for the entire dataset)
        self.num_foreground_voxels_for_intensitystats = num_foreground_voxels_for_intensitystats # Maybe increase it

    @staticmethod
    def collect_foreground_intensities(segmentation: np.ndarray, images: np.ndarray, seed: int = 1234,
                                       num_samples: int = 10000):
        """
        images=image with multiple channels = shape (c, x, y(, z))
        """
        assert images.ndim == 4 and segmentation.ndim == 4
        assert not np.any(np.isnan(segmentation)), "Segmentation contains NaN values. grrrr.... :-("
        assert not np.any(np.isnan(images)), "Images contains NaN values. grrrr.... :-("

        rs = np.random.RandomState(seed)

        intensities_per_channel = []
        # we don't use the intensity_statistics_per_channel at all, it's just something that might be nice to have
        intensity_statistics_per_channel = []

        # segmentation is 4d: 1,x,y,z. We need to remove the empty dimension for the following code to work
        foreground_mask = segmentation[0] > 0
        percentiles = np.array((0.5, 50.0, 99.5))

        for i in range(len(images)):
            foreground_pixels = images[i][foreground_mask]
            num_fg = len(foreground_pixels)
            # sample with replacement so that we don't get issues with cases that have less than num_samples
            # foreground_pixels. We could also just sample less in those cases but that would than cause these
            # training cases to be underrepresented
            intensities_per_channel.append(
                rs.choice(foreground_pixels, num_samples, replace=True) if num_fg > 0 else [])

            mean, median, mini, maxi, percentile_99_5, percentile_00_5 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            if num_fg > 0:
                percentile_00_5, median, percentile_99_5 = np.percentile(foreground_pixels, percentiles)
                mean = np.mean(foreground_pixels)
                mini = np.min(foreground_pixels)
                maxi = np.max(foreground_pixels)

            intensity_statistics_per_channel.append({
                'mean': mean,
                'median': median,
                'min': mini,
                'max': maxi,
                'percentile_99_5': percentile_99_5,
                'percentile_00_5': percentile_00_5,

            })

        return intensities_per_channel, intensity_statistics_per_channel

    @staticmethod
    def collect_input_image_intensities(images: np.ndarray):
        """
        images=image with multiple channels = shape (c, x, y(, z))
        """

        assert not np.any(np.isnan(images)), "Images contains NaN values... :-("

        # we don't use the intensity_statistics_per_channel at all, it's just something that might be nice to have
        intensity_statistics_per_channel = []

        # segmentation is 4d: 1,x,y,z. We need to remove the empty dimension for the following code to work

        percentiles = np.array((0.5, 1.0, 50.0, 99.0, 99.5))

        for i in range(len(images)):
            pixels = images[i]

            percentile_00_5, percentile_01_0, median, percentile_99_0, percentile_99_5 = np.percentile(pixels,
                                                                                                       percentiles)
            mean = np.mean(pixels)
            mini = np.min(pixels)
            maxi = np.max(pixels)

            intensity_statistics_per_channel.append({
                'mean': float(mean),
                'median': float(median),
                'min': float(mini),
                'max': float(maxi),
                'percentile_99_5': float(percentile_99_5),
                'percentile_99_0': float(percentile_99_0),
                'percentile_01_0': float(percentile_01_0),
                'percentile_00_5': float(percentile_00_5),
            })

        return intensity_statistics_per_channel

    @staticmethod
    def analyze_case(image_files: List[str], segmentation_file: str, reader_writer_class: Type[BaseReaderWriter],
                     num_samples: int = 10000, extract_input_image_statistics: bool = False):
        rw = reader_writer_class()
        images, properties_images = rw.read_images(image_files)
        if segmentation_file is not None:  # SSL some other variants don't have seg_file
            segmentation, properties_seg = rw.read_seg(segmentation_file)

            # we no longer crop and save the cropped images before this is run. Instead we run the cropping on the fly.
            # Downside is that we need to do this twice (once here and once during preprocessing). Upside is that we don't
            # need to save the cropped data anymore. Given that cropping is not too expensive it makes sense to do it this
            # way. This is only possible because we are now using our new input/output interface.
            try:
                data_cropped, seg_cropped, bbox = crop_to_nonzero(images, segmentation)
            except Exception as e:
                StringUtils.print(f"crop_to_nonzero: {e} \n{image_files}, {segmentation_file}")
                raise

            foreground_intensities_per_channel, foreground_intensity_stats_per_channel = \
                DatasetFingerprintExtractor.collect_foreground_intensities(seg_cropped, data_cropped,
                                                                           num_samples=num_samples)
        else:
            # set to None for SSL and other variants with no segments
            data_cropped, seg_cropped, bbox = crop_to_nonzero(images, seg=None)
            foreground_intensities_per_channel, foreground_intensity_stats_per_channel = DatasetFingerprintExtractor.collect_foreground_intensities(
                np.ones_like(data_cropped), data_cropped,
                num_samples=num_samples)

        if extract_input_image_statistics:
            intensity_statistics_per_channel = DatasetFingerprintExtractor.collect_input_image_intensities(data_cropped)
        else:
            intensity_statistics_per_channel = None

        spacing = properties_images['spacing']

        shape_before_crop = images.shape[1:]
        shape_after_crop = data_cropped.shape[1:]
        relative_size_after_cropping = np.prod(shape_after_crop) / np.prod(shape_before_crop)
        return (shape_after_crop, spacing, foreground_intensities_per_channel, foreground_intensity_stats_per_channel,
                relative_size_after_cropping, intensity_statistics_per_channel)

    def run(self, overwrite_existing: bool = False, continue_: bool = False,
            extract_input_image_statistics: bool = False, keys_percent: float = None) -> dict:
        # we do not save the properties file in self.input_folder because that folder might be read-only. We can only
        # reliably write in nnUNet_preprocessed and nnUNet_results, so nnUNet_preprocessed it is
        print(f"run DatasetFingerprintExtractor: {extract_input_image_statistics=}")
        preprocessed_output_folder = join(nnUNet_preprocessed, self.dataset_name)
        maybe_mkdir_p(preprocessed_output_folder)
        properties_file = join(preprocessed_output_folder, 'dataset_fingerprint.json')

        properties_file = DirUtils.split_extension(properties_file, suffix="_continue")
        if not isfile(properties_file) or overwrite_existing or continue_:
            if continue_ and isfile(properties_file):
                fingerprint = load_json(properties_file)
                done_keys = fingerprint["spacings"].keys()
                print(f"[INFO] Already {len(done_keys)} has been extracted...")
                keys = set(self.dataset.keys()) - set(done_keys)
            else:
                fingerprint = {
                    "spacings": dict(),
                    "shapes_after_crop": dict(),
                    "median_relative_size_after_cropping": None,
                    "relative_size_after_cropping": dict(),
                    "input_images_intensity_statistics_per_channel": dict(),
                    # "foreground_intensities_per_channel_per_sample": dict(),
                    "foreground_intensity_properties_per_channel": dict()
                }
                keys = self.dataset.keys()
                
            if keys_percent is not None:
                keys = list(keys)[:round(len(keys) * keys_percent)]
                
            reader_writer_class = determine_reader_writer_from_dataset_json(self.dataset_json,
                                                                            # yikes. Rip the following line
                                                                            self.dataset[
                                                                                self.dataset.keys().__iter__().__next__()][
                                                                                'images'][0])

            # determine how many foreground voxels we need to sample per training case
            num_foreground_samples_per_case = int(self.num_foreground_voxels_for_intensitystats //
                                                  len(self.dataset))

            r = []
            with multiprocessing.get_context("spawn").Pool(self.num_processes) as p:
                for k in tqdm(keys, desc="Adding all samples to analyze_case"):
                    r.append(p.starmap_async(DatasetFingerprintExtractor.analyze_case,
                                             ((self.dataset[k]['images'], self.dataset[k]['label'],
                                               reader_writer_class,
                                               num_foreground_samples_per_case,
                                               extract_input_image_statistics
                                               ),)))
                remaining = list(range(len(keys)))
                # p is pretty nifti. If we kill workers they just respawn but don't do any work.
                # So we need to store the original pool of workers.
                workers = [j for j in p._pool]
                with tqdm(desc="Extracting Fingerprints...", total=len(keys)) as pbar:
                    while len(remaining) > 0:
                        all_alive = all([j.is_alive() for j in workers])
                        if not all_alive:
                            raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
                                               'OK jokes aside.\n'
                                               'One of your background processes is missing. This could be because of '
                                               'an error (look for an error message) or because it was killed '
                                               'by your OS due to running out of RAM. If you don\'t see '
                                               'an error message, out of RAM is likely the problem. In that case '
                                               'reducing the number of workers might help')
                        done = [i for i in remaining if r[i].ready()]
                        for _ in done:
                            pbar.update()
                        remaining = [i for i in remaining if i not in done]
                        sleep(0.1)

            # results = ptqdm(DatasetFingerprintExtractor.analyze_case,
            #                 (training_images_per_case, training_labels_per_case),
            #                 processes=self.num_processes, zipped=True, reader_writer_class=reader_writer_class,
            #                 num_samples=num_foreground_samples_per_case, disable=self.verbose)
            results = [i.get()[0] for i in r]

            # Outputs of DatasetFingerprintExtractor.analyze_case:
            # shape_after_crop, spacing, foreground_intensities_per_channel, foreground_intensity_stats_per_channel, \
            #     relative_size_after_cropping

            shapes_after_crop = {k: r[0] for r, k in zip(results, keys)}
            spacings = {k: r[1] for r, k in zip(results, keys)}

            num_channels = len(self.dataset_json['channel_names'].keys()
                               if 'channel_names' in self.dataset_json.keys()
                               else self.dataset_json['modality'].keys())
            # foreground_intensity_properties_per_channel = fingerprint['foreground_intensity_properties_per_channel']
            # foreground_intensities_per_channel_per_sample = fingerprint['foreground_intensities_per_channel_per_sample']
            foreground_intensity_properties_per_channel = dict()
            if results and results[0][2] is not None:  # This is used for not SSL or anything with foreground
                foreground_intensities_per_channel_per_sample = ({k: [r[2][i] for i in range(num_channels)] for r, k in
                                                                  zip(results, keys)}
                    # | foreground_intensities_per_channel_per_sample
                    )
                foreground_intensities_per_channel = [
                    np.concatenate([r[i] for r in foreground_intensities_per_channel_per_sample.values()]) for i in
                    range(num_channels)]
                foreground_intensities_per_channel = np.array(foreground_intensities_per_channel) # shape: [1, num_foreground_samples_per_case* len(keys)]
                percentiles = np.array((0.5, 50.0, 99.5))
                for i in range(num_channels):
                    percentile_00_5, median, percentile_99_5 = np.percentile(foreground_intensities_per_channel[i], percentiles)
                    foreground_intensity_properties_per_channel[i] = {
                        'mean': float(np.mean(foreground_intensities_per_channel[i])),
                        'median': float(median),
                        'std': float(np.std(foreground_intensities_per_channel[i])),
                        'min': float(np.min(foreground_intensities_per_channel[i])),
                        'max': float(np.max(foreground_intensities_per_channel[i])),
                        'percentile_00_5': float(percentile_00_5),
                        'percentile_99_5': float(percentile_99_5),
                    }
                    if fingerprint['foreground_intensity_properties_per_channel']:
                        print(
                            "[WARNING] I am just meaning the inputs. Super Bad. Fix it later. Super Bad. I tried to save all the inputs in json file but the final json was Super Large :)")
                        foreground_intensity_properties_per_channel[i]['mean'] = (
                                                                                         foreground_intensity_properties_per_channel[
                                                                                             i]['mean'] +
                                                                                         fingerprint[
                                                                                             'foreground_intensity_properties_per_channel'][
                                                                                             i]['mean']) / 2
                        foreground_intensity_properties_per_channel[i]['median'] = (
                                                                                           foreground_intensity_properties_per_channel[
                                                                                               i]['median'] +
                                                                                           fingerprint[
                                                                                               'foreground_intensity_properties_per_channel'][
                                                                                               i]['median']) / 2
                        foreground_intensity_properties_per_channel[i]['std'] = (
                                                                                        foreground_intensity_properties_per_channel[
                                                                                            i]['std'] + fingerprint[
                                                                                            'foreground_intensity_properties_per_channel'][
                                                                                            i]['std']) / 2
                        foreground_intensity_properties_per_channel[i]['min'] = (
                                                                                        foreground_intensity_properties_per_channel[
                                                                                            i]['min'] + fingerprint[
                                                                                            'foreground_intensity_properties_per_channel'][
                                                                                            i]['min']) / 2
                        foreground_intensity_properties_per_channel[i]['max'] = (
                                                                                        foreground_intensity_properties_per_channel[
                                                                                            i]['max'] + fingerprint[
                                                                                            'foreground_intensity_properties_per_channel'][
                                                                                            i]['max']) / 2
                        foreground_intensity_properties_per_channel[i]['percentile_00_5'] = (
                                                                                                    foreground_intensity_properties_per_channel[
                                                                                                        i][
                                                                                                        'percentile_00_5'] +
                                                                                                    fingerprint[
                                                                                                        'foreground_intensity_properties_per_channel'][
                                                                                                        i][
                                                                                                        'percentile_00_5']) / 2
                        foreground_intensity_properties_per_channel[i]['percentile_99_5'] = (
                                                                                                    foreground_intensity_properties_per_channel[
                                                                                                        i][
                                                                                                        'percentile_99_5'] +
                                                                                                    fingerprint[
                                                                                                        'foreground_intensity_properties_per_channel'][
                                                                                                        i][
                                                                                                        'percentile_99_5']) / 2

                # make json friendly
                # foreground_intensities_per_channel_per_sample = {k: [item.tolist() for item in v_list] for k, v_list in foreground_intensities_per_channel_per_sample.items()}
            # Get input image percentile:

            # we drop this so that the json file is somewhat human-readable
            # foreground_intensity_stats_by_case_and_modality = [r[3] for r in results]
            # median_relative_size_after_cropping = np.median([r[4] for r in results], 0)
            relative_size_after_cropping = {k: r[4] for r, k in zip(results, keys)}

            input_images_intensity_statistics_per_channel = {k: r[5] for r, k in zip(results,
                                                                                     keys)} if extract_input_image_statistics else dict()

            fingerprint = {
                "spacings": spacings | fingerprint["spacings"],
                "shapes_after_crop": shapes_after_crop | fingerprint["shapes_after_crop"],
                'foreground_intensity_properties_per_channel': foreground_intensity_properties_per_channel,
                "median_relative_size_after_cropping": float(np.median(
                    list((relative_size_after_cropping | fingerprint["relative_size_after_cropping"]).values()))),
                "relative_size_after_cropping": relative_size_after_cropping | fingerprint[
                    "relative_size_after_cropping"],
                "input_images_intensity_statistics_per_channel": input_images_intensity_statistics_per_channel |
                                                                 fingerprint[
                                                                     "input_images_intensity_statistics_per_channel"],
                # "foreground_intensities_per_channel_per_sample": foreground_intensities_per_channel_per_sample
            }
            node_rank = os.environ.get("SLURM_NODEID", "0")
            if node_rank == 0:
                if isfile(properties_file):
                    os.remove(properties_file)
                    save_json(fingerprint, properties_file)
                else:
                    save_json(fingerprint, properties_file)

            
            if extract_input_image_statistics:
                print(f"saved in {DirUtils.split_extension(properties_file, suffix="_populated")}")
                save_json(fingerprint, DirUtils.split_extension(properties_file, suffix="_populated"))
            # Saving the less populated version
            main_properties_file = StringUtils.right_replace(properties_file, "_continue", "", count=1)
            if main_properties_file != properties_file:
                fingerprint = {
                    "spacings": list(fingerprint["spacings"].values()),
                    "shapes_after_crop": list(fingerprint["shapes_after_crop"].values()),
                    "median_relative_size_after_cropping": fingerprint["median_relative_size_after_cropping"],
                    "foreground_intensity_properties_per_channel": fingerprint["foreground_intensity_properties_per_channel"],
                }
                save_json(fingerprint, main_properties_file)
        else:
            fingerprint = load_json(properties_file)
        return fingerprint


if __name__ == '__main__':
    dfe = DatasetFingerprintExtractor(2, 8)
    dfe.run(overwrite_existing=False)
