from typing import Tuple, Union, List
import random

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.preprocessing.normalization. default_normalization_schemes import CTNormalization


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k) if label is not None else None
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy() if label is not None else None
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False) if label is not None else None
    return image, label


class RandomGenerator:
    def __init__(self, output_size: Tuple[int, ...], val: bool = False):
        self.output_size = output_size
        self.val = val

    def __call__(self, sample):
        image, label = sample['image'], sample.get('label')

        if random.random() > 0.5 and not self.val:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5 and not self.val:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y),
                         order=0) if label is not None else None
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32)) if label is not None else None
        if label is not None:
            sample = {'image': image, 'label': label.long()}
        else:
            sample = {'image': image}
        return sample


class SSLMAERandomAug(AbstractTransform):
    def __init__(self, patch_size, key_origin="image", key_target="image", val: bool=False):
        self.key_target = key_target
        self.key_origin = key_origin
        self.transform = RandomGenerator(patch_size, val=val)


    def __call__(self, **data_dict):
        data = data_dict[self.key_origin][0].numpy().copy().astype(np.float32)
        sample = {'image': data}
        data = self.transform(sample)['image'].unsqueeze(0)
        data_dict[self.key_target] = data
        return data_dict


class SSLNormalizationAug(AbstractTransform):

    def __init__(self, key_origin="image", key_target="image"):
        self.key_target = key_target
        self.key_origin = key_origin


    def __call__(self, **data_dict):
        if "image" in data_dict:
            data = data_dict["image"][0]
        elif "data" in data_dict:
            data = data_dict['data'][0]
        else:
            raise ValueError(f"{data_dict.keys()} :( I'm stuck in {self.__class__.__name__}")
        mean = data.mean()
        std = data.std()
        data = (data - mean) / (max(std, 1e-8))
        data = data.unsqueeze(0)
        data_dict[self.key_target] = data
        return data_dict
    
    
class FullPreprocessing:
    def __init__(self, dataset_json, plans_manager, configuration_manager, normalization_cls=CTNormalization, spacing=None, verbose=False):
        
        self.verbose=verbose
        self.dataset_json=dataset_json
        self.plans_manager = plans_manager
        self.plans_manager_transpose_forward=self.plans_manager.transpose_forward
        self.configuration_manager_spacing= spacing or configuration_manager.spacing
        self.normalization_cls = normalization_cls
        self.configuration_manager=configuration_manager
        self.configuration_manager.configuration['resampling_fn_data_kwargs'].update({"verbose": False})
        self.configuration_manager_resampling_fn_data=self.configuration_manager.resampling_fn_data
        self.plans_manager_foreground_intensity_properties_per_channel = plans_manager.foreground_intensity_properties_per_channel
    
    
    @staticmethod
    def _sample_foreground_locations(seg: np.ndarray, classes_or_regions: Union[List[int], List[Tuple[int, ...]]],
                                     seed: int = 1234, verbose: bool = False):
        num_samples = 10000
        min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too
        # sparse
        rndst = np.random.RandomState(seed)
        class_locs = {}
        for c in classes_or_regions:
            k = c if not isinstance(c, list) else tuple(c)
            if isinstance(c, (tuple, list)):
                mask = seg == c[0]
                for cc in c[1:]:
                    mask = mask | (seg == cc)
                all_locs = np.argwhere(mask)
            else:
                all_locs = np.argwhere(seg == c)
            if len(all_locs) == 0:
                class_locs[k] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

            selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
            class_locs[k] = selected
            if verbose:
                print(c, target_num_samples)
        return class_locs
    def __call__(self, data_dict):
        
        if "image" in data_dict:
            data = data_dict["image"]
        elif "data" in data_dict:
            data = data_dict['data']
        else:
            raise KeyError(f"{data_dict.keys()} :( I'm stuck in {self.__class__.__name__}")
        
        seg = data_dict.get("seg", None)
        # set verbose to None
        
        properties = data_dict["properties"]
        # def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
        #              plans_manager: PlansManager, configuration_manager: ConfigurationManager,
        #              dataset_json: Union[dict, str]):
        # let's not mess up the inputs!
        data = np.copy(data)
        if seg is not None:
            assert data.shape[1:] == seg.shape[
                1:], "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            seg = np.copy(seg)

        has_seg = seg is not None
        
        reduce = False
        # apply transpose_forward, this also needs to be applied to the spacing!
        if len(data.shape) == len(self.plans_manager_transpose_forward):
            reduce = True
            data = data[None, ...]

        data = data.transpose([0, *[i + 1 for i in self.plans_manager_transpose_forward]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in self.plans_manager_transpose_forward]])
        original_spacing = [properties['spacing'][i] for i in self.plans_manager_transpose_forward]
        
        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping
        
        if has_seg:
            data, seg, bbox = crop_to_nonzero(data, seg)
        else:
            data, _, bbox = crop_to_nonzero(data)
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, _, bbox = crop_to_nonzero(data)
        properties['bbox_used_for_cropping'] = bbox
        # print(data.shape, seg.shape)
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # resample

        target_spacing = self.configuration_manager_spacing  # this should already be transposed
        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 2d configuration we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        if -1 not in target_spacing:
            new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)
        else:
            target_spacing = original_spacing
            new_shape = data.shape[1:]
        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        data = self._normalize(data)

        # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        # old_shape = data.shape[1:]
        # tic = time.time()
        if original_spacing != target_spacing:
            data = self.configuration_manager_resampling_fn_data(data, new_shape, original_spacing, target_spacing, verbose=False)

        if np.isnan(data).any():
            print(f"nan seen in the input: {data_dict['index']}")
        # if we have a segmentation, sample foreground locations for oversampling and add those to properties
        if has_seg:
            if original_spacing != target_spacing:
                seg = self.configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing, verbose=False)
        if has_seg:
            # reinstantiating LabelManager for each case is not ideal. We could replace the dataset_json argument
            # with a LabelManager Instance in this function because that's all its used for. Dunno what's better.
            # LabelManager is pretty light computation-wise.
            label_manager = self.plans_manager.get_label_manager(self.dataset_json)
            collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                else label_manager.foreground_labels

            # when using the ignore label we want to sample only from annotated regions. Therefore we also need to
            # collect samples uniformly from all classes (incl background)
            if label_manager.has_ignore_label:
                collect_for_this.append(label_manager.all_labels)

            # no need to filter background in regions because it is already filtered in handle_labels
            # print(all_labels, regions)
            properties['class_locations'] = self._sample_foreground_locations(seg, collect_for_this,
                                                                              verbose=self.verbose)
            # seg = self.modify_seg_fn(seg, self.plans_manager, self.dataset_json, self.configuration_manager)
        if has_seg:
            if np.max(seg) > 127:
                seg = seg.astype(np.int16)
            else:
                seg = seg.astype(np.int8)
        
        if reduce:
            return data[0], seg
        else:
            return data, seg
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        if self.normalization_cls is not None:
            for c in range(data.shape[0]):
                normalizer = self.normalization_cls(use_mask_for_norm=None, intensityproperties=self.plans_manager_foreground_intensity_properties_per_channel.get(str(c), dict()))
                data[c] = normalizer.run(data[c], None)
        return data