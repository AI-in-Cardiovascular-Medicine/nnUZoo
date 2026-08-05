import torch
import math
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from os.path import split
from typing import List, Union, Type, Tuple

import blosc2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile, write_pickle
from deep_utils import StringUtils, JsonUtils
from nnunetv2.configuration import default_num_processes
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.training.dataloading.utils import unpack_dataset


class nnUNetBaseDataset(ABC):
    """
    Defines the interface
    """

    def __init__(self, folder: str, identifiers: List[str] = None,
                 folder_with_segs_from_previous_stage: str = None,
                 target_type: str = "segmentation",
                 **kwargs):
        super().__init__()
        # print('loading dataset')
        if identifiers is None:
            identifiers = self.get_identifiers(folder, target_type)
        else:
            print("identifiers are given")
        identifiers.sort()
        self.source_folder = folder
        self.folder_with_segs_from_previous_stage = folder_with_segs_from_previous_stage
        self.identifiers = identifiers
        self.target_type = target_type
        self.dataset_json = JsonUtils.load(join(split(folder.replace(str(nnUNet_preprocessed).strip("/"), str(nnUNet_raw).strip("/")))[0], "dataset.json"))
        self.kwargs = kwargs

    def __getitem__(self, identifier):
        return self.load_case(identifier)

    @abstractmethod
    def load_case(self, identifier):
        pass

    @staticmethod
    @abstractmethod
    def save_case(
            data: np.ndarray,
            seg: np.ndarray,
            properties: dict,
            output_filename_truncated: str
    ):
        pass

    @staticmethod
    @abstractmethod
    def get_identifiers(folder: str, target_type: str) -> List[str]:
        pass

    @staticmethod
    def unpack_dataset(folder: str, overwrite_existing: bool = False,
                       num_processes: int = default_num_processes,
                       verify: bool = True):
        pass

    def __len__(self):
        return len(self.identifiers)


class nnUNetDatasetNiiGz(nnUNetBaseDataset):
    def load_case(self, identifier):
        identifier = str(identifier)
        nnUNet_preprocessed = str(nnUNet_preprocessed)
        nnUNet_raw = str(nnUNet_raw)
        if nnUNet_preprocessed in self.source_folder:
            folder = self.source_folder.replace(nnUNet_preprocessed, nnUNet_raw)
            folder = join(split(folder)[0], "imagesTr")
            image_files = []
            for ii in range(len(self.dataset_json["channel_names"])):
                filepath = join(folder, identifier + f'_{ii:04}.nii.gz')
                if os.path.exists(filepath):
                    image_files.append(filepath)
                else:
                    raise FileNotFoundError(filepath)


            if self.target_type in ["segmentation"]:
                seg_file = image_files[0].replace("_0000.nii.gz", ".nii.gz").replace("imagesTr", "labelsTr")
                seg, _ = rw.read_images([seg_file])
            # seg = seg[0] 
            elif self.target_type in ['ssl']:
                seg = None
            else:
                raise ValueError()
        else:
            image_files = []
            for ii in range(len(self.dataset_json["channel_names"])):
                filepath = join(folder, identifier + f'.nii.gz')
                if os.path.exists(filepath):
                    image_files.append(filepath)
                else:
                    raise FileNotFoundError(filepath)
            seg = None # Test mode

        rw = SimpleITKIO()
        data, properties = rw.read_images(image_files)
        # data, _, bbox = crop_to_nonzero(data)

        properties['shape_before_cropping'] = data.shape
        properties['shape_after_cropping_and_before_resampling'] = data.shape
        properties['bbox_used_for_cropping'] = [[0, data.shape[0]], [0, data.shape[1]], [0, data.shape[2]]]

        seg_prev = None
        if np.isnan(data).any():
            print(f"nan seen in the input: {identifier}")
        
        return data, seg, seg_prev, properties
    
        
    @staticmethod
    def save_case(
            data: np.ndarray,
            seg: np.ndarray,
            properties: dict,
            output_filename_truncated: str
    ):
        np.savez_compressed(output_filename_truncated + '.npz', data=data, seg=seg)
        write_pickle(properties, output_filename_truncated + '.pkl')

    @staticmethod
    def save_seg(
            seg: np.ndarray,
            output_filename_truncated: str
    ):
        np.savez_compressed(output_filename_truncated + '.npz', seg=seg)

    @staticmethod
    def get_identifiers(folder: str, target_type: str) -> List[str]:
        """
        returns all identifiers in the preprocessed data folder
        """
        # folder = folder.replace(nnUNet_preprocessed, nnUNet_raw)
        # case_identifiers = list(JsonUtils.load(join(split(folder)[0], "dataset.json"))['train_labels'].keys())
        # case_identifiers = list(set([i[:-12] for i in os.listdir(folder) if i.endswith(".nii.gz")]))
        dataset_json = JsonUtils.load(join(split(folder.replace(nnUNet_preprocessed, nnUNet_raw))[0], "dataset.json")) # use the dataset in the nnunet_raw instead of nnunet_preprocessed
        target_type = dataset_json.get("target_type", "segmetnation")
        if target_type in ['classification']:
            case_identifiers = list(dataset_json['train_labels'].keys())
        else:
            case_identifiers = list(set([i[:-12] for i in os.listdir(folder) if i.endswith(".nii.gz")]))
        return case_identifiers

    @staticmethod
    def unpack_dataset(folder: str, overwrite_existing: bool = False,
                       num_processes: int = default_num_processes,
                       verify: bool = True,
                       unpack_segmentation=True, ):
        # return unpack_dataset(folder, unpack_segmentation, overwrite_existing, num_processes, verify)
        StringUtils.print("There are no preprocessed data to unpack")


class nnUNetDatasetNumpy(nnUNetBaseDataset):
    def load_case(self, identifier:str): #  if it's the case number take the identifier and go on :)
        data_npy_file = join(self.source_folder, identifier + '.npy')
        if not isfile(data_npy_file):
            try:
                data_path = join(self.source_folder, identifier + '.npz')
                data = np.load(data_path)['data']
            except Exception as e:
                StringUtils.print(f"[ERROR] Not able to load file: {data_path} with error: {e}")
                raise
        else:
            try:
                data = np.load(data_npy_file, mmap_mode='r')
            except Exception as e:
                StringUtils.print(f"[ERROR] Not able to load file: {data_npy_file}  with error: {e}")
                raise
        if self.target_type in ["segmentation", "translation"]:
            seg_npy_file = join(self.source_folder, identifier + '_seg.npy')
            if not isfile(seg_npy_file):
                seg = np.load(join(self.source_folder, identifier + '.npz'))['seg']
            else:
                seg = np.load(seg_npy_file, mmap_mode='r')
        elif self.target_type in ["classification"]:
            seg = None
            y_true = self.dataset_json['train_labels'][str(identifier)]
        else:
            seg = None
        if self.folder_with_segs_from_previous_stage is not None:
            prev_seg_npy_file = join(self.folder_with_segs_from_previous_stage, identifier + '.npy')
            if isfile(prev_seg_npy_file):
                seg_prev = np.load(prev_seg_npy_file, 'r')
            else:
                seg_prev = np.load(join(self.folder_with_segs_from_previous_stage, identifier + '.npz'))['seg']
        else:
            seg_prev = None
        properties = load_pickle(join(self.source_folder, identifier + '.pkl'))
        if self.target_type in ['classification']:
            return data, properties, y_true
        return data, seg, seg_prev, properties

    @staticmethod
    def save_case(
            data: np.ndarray,
            seg: np.ndarray,
            properties: dict,
            output_filename_truncated: str
    ):
        np.savez_compressed(output_filename_truncated + '.npz', data=data, seg=seg)
        write_pickle(properties, output_filename_truncated + '.pkl')

    @staticmethod
    def save_seg(
            seg: np.ndarray,
            output_filename_truncated: str
    ):
        np.savez_compressed(output_filename_truncated + '.npz', seg=seg)

    @staticmethod
    def get_identifiers(folder: str, target_type: str) -> List[str]:
        """
        returns all identifiers in the preprocessed data folder
        """
        if target_type in ['segmentation', "translation"]:
            case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith(".npz")]
        elif target_type in ['classification']:
            case_identifiers = list(JsonUtils.load(join(split(folder)[0], "dataset.json"))['train_labels'].keys())
        else:
            case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith(".npy")]
        return case_identifiers

    @staticmethod
    def unpack_dataset(folder: str, overwrite_existing: bool = False,
                       num_processes: int = default_num_processes,
                       verify: bool = True,
                       unpack_segmentation=True, ):
        return unpack_dataset(folder, unpack_segmentation, overwrite_existing, num_processes, verify)

    
    # def __getitem__(self, identifier):
    #     if isinstance(identifier, int):
    #         identifier = self.identifiers[identifier]
    #         if self.target_type in ['classification']:
    #             data, properties, y_true = self.load_case(identifier)
    #         else:
    #             raise NotImplementedError("ha ha ha")
    #         properties['keys'] = identifier
    #         transform = self.kwargs.get("transform", None)
    #         if transform:
    #             if self.dict_input:
    #                 data = transform({"image": torch.from_numpy(data.copy()), "properties": properties})['image']
    #             else:
    #                 data = transform(torch.from_numpy(data.copy()))
    #         return data, properties, y_true
    #     else:
    #         return self.load_case(identifier)

    
class nnUNetDatasetMonaiNumpy(nnUNetBaseDataset):
    def __init__(self, dict_input, *args, **kwargs):
        self.dict_input = dict_input
        super().__init__(*args, **kwargs)
    
    def load_case(self, identifier:str): #  if it's the case number take the identifier and go on :)
        data_npy_file = join(self.source_folder, identifier + '.npy')
        if not isfile(data_npy_file):
            try:
                data_path = join(self.source_folder, identifier + '.npz')
                data = np.load(data_path)['data']
            except Exception as e:
                StringUtils.print(f"[ERROR] Not able to load file: {data_path} with error: {e}")
                raise
        else:
            try:
                data = np.load(data_npy_file, mmap_mode='r')
            except Exception as e:
                StringUtils.print(f"[ERROR] Not able to load file: {data_npy_file}  with error: {e}")
                raise
        if self.target_type in ["segmentation", "translation"]:
            seg_npy_file = join(self.source_folder, identifier + '_seg.npy')
            if not isfile(seg_npy_file):
                seg = np.load(join(self.source_folder, identifier + '.npz'))['seg']
            else:
                seg = np.load(seg_npy_file, mmap_mode='r')
        elif self.target_type in ["classification"]:
            seg = None
            y_true = self.dataset_json['train_labels'][str(identifier)]
        else:
            seg = None
        if self.folder_with_segs_from_previous_stage is not None:
            prev_seg_npy_file = join(self.folder_with_segs_from_previous_stage, identifier + '.npy')
            if isfile(prev_seg_npy_file):
                seg_prev = np.load(prev_seg_npy_file, 'r')
            else:
                seg_prev = np.load(join(self.folder_with_segs_from_previous_stage, identifier + '.npz'))['seg']
        else:
            seg_prev = None
        properties = load_pickle(join(self.source_folder, identifier + '.pkl'))
        if self.target_type in ['classification']:
            return data, properties, y_true
        return data, seg, seg_prev, properties

    @staticmethod
    def save_case(
            data: np.ndarray,
            seg: np.ndarray,
            properties: dict,
            output_filename_truncated: str
    ):
        np.savez_compressed(output_filename_truncated + '.npz', data=data, seg=seg)
        write_pickle(properties, output_filename_truncated + '.pkl')

    @staticmethod
    def save_seg(
            seg: np.ndarray,
            output_filename_truncated: str
    ):
        np.savez_compressed(output_filename_truncated + '.npz', seg=seg)

    @staticmethod
    def get_identifiers(folder: str, target_type: str) -> List[str]:
        """
        returns all identifiers in the preprocessed data folder
        """
        if target_type in ['segmentation', "translation"]:
            case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith(".npz")]
        elif target_type in ['classification']:
            case_identifiers = list(JsonUtils.load(join(split(folder)[0], "dataset.json"))['train_labels'].keys())
        else:
            case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith(".npy")]
        return case_identifiers

    @staticmethod
    def unpack_dataset(folder: str, overwrite_existing: bool = False,
                       num_processes: int = default_num_processes,
                       verify: bool = True,
                       unpack_segmentation=True, ):
        return unpack_dataset(folder, unpack_segmentation, overwrite_existing, num_processes, verify)

    
    def __getitem__(self, identifier):
        
        identifier = self.identifiers[identifier]
        if self.target_type in ['classification']:
            data, properties, y_true = self.load_case(identifier)
        else:
            raise NotImplementedError("ha ha ha")
        properties['keys'] = identifier
        transform = self.kwargs.get("transform", None)
        if transform:
            if self.dict_input:
                data = transform({"image": torch.from_numpy(data.copy()), "properties": properties})['image']
            else:
                data = transform(torch.from_numpy(data.copy()))
        return data, properties, y_true
        

class nnUNetDatasetBlosc2(nnUNetBaseDataset):
    def __init__(self, folder: str, identifiers: List[str] = None,
                 folder_with_segs_from_previous_stage: str = None,
                 target_type: str = "segmentation"):
        super().__init__(folder, identifiers, folder_with_segs_from_previous_stage, target_type=target_type)
        blosc2.set_nthreads(1)

    def __getitem__(self, identifier):
        return self.load_case(identifier)

    def load_case(self, identifier):
        dparams = {
            'nthreads': 1
        }
        data_b2nd_file = join(self.source_folder, identifier + '.b2nd')

        # mmap does not work with Windows -> https://github.com/MIC-DKFZ/nnUNet/issues/2723
        mmap_kwargs = {} if os.name == "nt" else {'mmap_mode': 'r'}
        data = blosc2.open(urlpath=data_b2nd_file, mode='r', dparams=dparams, **mmap_kwargs)

        seg_b2nd_file = join(self.source_folder, identifier + '_seg.b2nd')
        seg = blosc2.open(urlpath=seg_b2nd_file, mode='r', dparams=dparams, **mmap_kwargs)

        if self.folder_with_segs_from_previous_stage is not None:
            prev_seg_b2nd_file = join(self.folder_with_segs_from_previous_stage, identifier + '.b2nd')
            seg_prev = blosc2.open(urlpath=prev_seg_b2nd_file, mode='r', dparams=dparams, **mmap_kwargs)
        else:
            seg_prev = None

        properties = load_pickle(join(self.source_folder, identifier + '.pkl'))
        return data, seg, seg_prev, properties

    @staticmethod
    def save_case(
            data: np.ndarray,
            seg: np.ndarray,
            properties: dict,
            output_filename_truncated: str,
            chunks=None,
            blocks=None,
            chunks_seg=None,
            blocks_seg=None,
            clevel: int = 8,
            codec=blosc2.Codec.ZSTD
    ):
        blosc2.set_nthreads(1)
        if chunks_seg is None:
            chunks_seg = chunks
        if blocks_seg is None:
            blocks_seg = blocks

        cparams = {
            'codec': codec,
            # 'filters': [blosc2.Filter.SHUFFLE],
            # 'splitmode': blosc2.SplitMode.ALWAYS_SPLIT,
            'clevel': clevel,
        }
        # print(output_filename_truncated, data.shape, seg.shape, blocks, chunks, blocks_seg, chunks_seg, data.dtype, seg.dtype)
        blosc2.asarray(np.ascontiguousarray(data), urlpath=output_filename_truncated + '.b2nd', chunks=chunks,
                       blocks=blocks, cparams=cparams)
        blosc2.asarray(np.ascontiguousarray(seg), urlpath=output_filename_truncated + '_seg.b2nd', chunks=chunks_seg,
                       blocks=blocks_seg, cparams=cparams)
        write_pickle(properties, output_filename_truncated + '.pkl')

    @staticmethod
    def save_seg(
            seg: np.ndarray,
            output_filename_truncated: str,
            chunks_seg=None,
            blocks_seg=None
    ):
        blosc2.asarray(seg, urlpath=output_filename_truncated + '.b2nd', chunks=chunks_seg, blocks=blocks_seg)

    @staticmethod
    def get_identifiers(folder: str, target_type: str) -> List[str]:
        """
        returns all identifiers in the preprocessed data folder
        """
        case_identifiers = [i[:-5] for i in os.listdir(folder) if i.endswith(".b2nd") and not i.endswith("_seg.b2nd")]
        return case_identifiers

    @staticmethod
    def unpack_dataset(folder: str, overwrite_existing: bool = False,
                       num_processes: int = default_num_processes,
                       verify: bool = True, unpack_segmentation: bool = True):
        pass

    @staticmethod
    def comp_blosc2_params(
            image_size: Tuple[int, int, int, int],
            patch_size: Union[Tuple[int, int], Tuple[int, int, int]],
            bytes_per_pixel: int = 4,  # 4 byte are float32
            l1_cache_size_per_core_in_bytes=32768,  # 1 Kibibyte (KiB) = 2^10 Byte;  32 KiB = 32768 Byte
            l3_cache_size_per_core_in_bytes=1441792,
            # 1 Mibibyte (MiB) = 2^20 Byte = 1.048.576 Byte; 1.375MiB = 1441792 Byte
            safety_factor: float = 0.8  # we dont will the caches to the brim. 0.8 means we target 80% of the caches
    ):
        """
        Computes a recommended block and chunk size for saving arrays with blosc v2.

        Bloscv2 NDIM doku: "Remember that having a second partition means that we have better flexibility to fit the
        different partitions at the different CPU cache levels; typically the first partition (aka chunks) should
        be made to fit in L3 cache, whereas the second partition (aka blocks) should rather fit in L2/L1 caches
        (depending on whether compression ratio or speed is desired)."
        (https://www.blosc.org/posts/blosc2-ndim-intro/)
        -> We are not 100% sure how to optimize for that. For now we try to fit the uncompressed block in L1. This
        might spill over into L2, which is fine in our books.

        Note: this is optimized for nnU-Net dataloading where each read operation is done by one core. We cannot use threading

        Cache default values computed based on old Intel 4110 CPU with 32K L1, 128K L2 and 1408K L3 cache per core.
        We cannot optimize further for more modern CPUs with more cache as the data will need be be read by the
        old ones as well.

        Args:
            patch_size: Image size, must be 4D (c, x, y, z). For 2D images, make x=1
            patch_size: Patch size, spatial dimensions only. So (x, y) or (x, y, z)
            bytes_per_pixel: Number of bytes per element. Example: float32 -> 4 bytes
            l1_cache_size_per_core_in_bytes: The size of the L1 cache per core in Bytes.
            l3_cache_size_per_core_in_bytes: The size of the L3 cache exclusively accessible by each core. Usually the global size of the L3 cache divided by the number of cores.

        Returns:
            The recommended block and the chunk size.
        """
        # Fabians code is ugly, but eh

        num_channels = image_size[0]
        if len(patch_size) == 2:
            patch_size = [1, *patch_size]
        patch_size = np.array(patch_size)
        block_size = np.array((num_channels, *[2 ** (max(0, math.ceil(math.log2(i)))) for i in patch_size]))

        # shrink the block size until it fits in L1
        estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
        while estimated_nbytes_block > (l1_cache_size_per_core_in_bytes * safety_factor):
            # pick largest deviation from patch_size that is not 1
            axis_order = np.argsort(block_size[1:] / patch_size)[::-1]
            idx = 0
            picked_axis = axis_order[idx]
            while block_size[picked_axis + 1] == 1 or block_size[picked_axis + 1] == 1:
                idx += 1
                picked_axis = axis_order[idx]
            # now reduce that axis to the next lowest power of 2
            block_size[picked_axis + 1] = 2 ** (max(0, math.floor(math.log2(block_size[picked_axis + 1] - 1))))
            block_size[picked_axis + 1] = min(block_size[picked_axis + 1], image_size[picked_axis + 1])
            estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel

        block_size = np.array([min(i, j) for i, j in zip(image_size, block_size)])

        # note: there is no use extending the chunk size to 3d when we have a 2d patch size! This would unnecessarily
        # load data into L3
        # now tile the blocks into chunks until we hit image_size or the l3 cache per core limit
        chunk_size = deepcopy(block_size)
        estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
        while estimated_nbytes_chunk < (l3_cache_size_per_core_in_bytes * safety_factor):
            if patch_size[0] == 1 and all([i == j for i, j in zip(chunk_size[2:], image_size[2:])]):
                break
            if all([i == j for i, j in zip(chunk_size, image_size)]):
                break
            # find axis that deviates from block_size the most
            axis_order = np.argsort(chunk_size[1:] / block_size[1:])
            idx = 0
            picked_axis = axis_order[idx]
            while chunk_size[picked_axis + 1] == image_size[picked_axis + 1] or patch_size[picked_axis] == 1:
                idx += 1
                picked_axis = axis_order[idx]
            chunk_size[picked_axis + 1] += block_size[picked_axis + 1]
            chunk_size[picked_axis + 1] = min(chunk_size[picked_axis + 1], image_size[picked_axis + 1])
            estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
            if np.mean([i / j for i, j in zip(chunk_size[1:], patch_size)]) > 1.5:
                # chunk size should not exceed patch size * 1.5 on average
                chunk_size[picked_axis + 1] -= block_size[picked_axis + 1]
                break
        # better safe than sorry
        chunk_size = [min(i, j) for i, j in zip(image_size, chunk_size)]

        # print(image_size, chunk_size, block_size)
        return tuple(block_size), tuple(chunk_size)


file_ending_dataset_mapping = {
    'npz': nnUNetDatasetNumpy,
    'b2nd': nnUNetDatasetBlosc2,
    "nii.gz": nnUNetDatasetNiiGz,
    "gz": nnUNetDatasetNiiGz,
}



class MonaiDatasetNumpyNiiGz(nnUNetDatasetNumpy):
    def __init__(self, dict_input, *args, **kwargs):
        self.dict_input = dict_input
        super().__init__(*args, **kwargs)

    def load_case(self, identifier):
        identifier = str(identifier)
        if str(nnUNet_preprocessed) in str(self.source_folder):
            folder = self.source_folder.replace(str(nnUNet_preprocessed), str(nnUNet_raw))
            folder = join(split(folder)[0], "imagesTr")
        else:
            folder = self.source_folder
        if folder.endswith("imagesTr"):
            dataset_json_key="train_labels"
        elif folder.endswith("imagesTs"):
            dataset_json_key="test_labels"
        else:
            raise ValueError("only imagesTr and imagesTs is supported")
        
        image_files = []
        for ii in range(len(self.dataset_json["channel_names"])):
            filepath = join(folder, identifier + f'_{ii:04}.nii.gz')
            if os.path.exists(filepath):
                image_files.append(filepath)
            else:
                if not len(image_files):
                    raise ValueError(f"Didn't find any case for id: {identifier}")
                break
        
        rw = SimpleITKIO()
        data, properties = rw.read_images(image_files)
        # data, _, bbox = crop_to_nonzero(data)

        properties['shape_before_cropping'] = data.shape
        properties['shape_after_cropping_and_before_resampling'] = data.shape
        properties['bbox_used_for_cropping'] = [[0, data.shape[0]], [0, data.shape[1]], [0, data.shape[2]]]
        if dataset_json_key in self.dataset_json:
            y_true = self.dataset_json[dataset_json_key][str(identifier)]
        else:
            y_true  = None
        seg, seg_prev = None, None
        if np.isnan(data).any():
            print(f"nan seen in the input: {identifier}")
        return data, seg, seg_prev, properties, y_true
    
    
    def load_exact_case(self, identifier):
        image_files = [identifier] if isinstance(identifier, str) else identifier
        
        rw = SimpleITKIO()
        data, properties = rw.read_images(image_files)
        # data, _, bbox = crop_to_nonzero(data)

        properties['shape_before_cropping'] = data.shape
        properties['shape_after_cropping_and_before_resampling'] = data.shape
        properties['bbox_used_for_cropping'] = [[0, data.shape[0]], [0, data.shape[1]], [0, data.shape[2]]]
        if 'train_labels' in self.dataset_json:
            y_true = self.dataset_json['train_labels'][str(identifier)]
        else:
            y_true  = None
        seg, seg_prev = None, None
        if np.isnan(data).any():
            print(f"nan seen in the input: {identifier}")
        return data, seg, seg_prev, properties, y_true
    
    def __getitem__(self, idx):
        identifier = self.identifiers[idx]
        data, _, _, properties, y_true = self.load_case(identifier)
        properties['keys'] = identifier
        transform = self.kwargs.get("transform", None)
        if transform:
            if self.dict_input:
                data = transform({"image": torch.from_numpy(data.copy()), "properties": properties})['image']
            else:
                data = transform(torch.from_numpy(data.copy()))
        return data, properties, y_true

    
def infer_dataset_class(folder: str) -> Union[Type[nnUNetDatasetBlosc2], Type[nnUNetDatasetNumpy]]:
    try:
        from deep_utils import DirUtils
        # file_endings = set([os.path.basename(i).split('.')[-1] for i in subfiles(folder, join=False)])
        file_endings = set(["nii.gz" if item.endswith(".nii.gz") else item.split(".")[-1] for item in DirUtils.list_dir_full_path(folder, get_full_path=False)])
    except FileNotFoundError:
        raise ValueError(f"folder: {folder} not found!")
    except:
        return file_ending_dataset_mapping["nii.gz"]
    numpy_format = False
    if 'pkl' in file_endings:
        file_endings.remove('pkl')
    if 'npy' in file_endings:
        file_endings.remove('npy')
        numpy_format = True
    if len(file_endings) == 0 and numpy_format:
        return file_ending_dataset_mapping["npz"]

    assert len(file_endings) == 1, (f'Found more {len(file_endings)} file ending(s) in the folder {folder}. '
                                    f'endings: {file_endings}. '
                                    f'Unable to infer nnUNetDataset variant!')
    return file_ending_dataset_mapping[list(file_endings)[0]]
