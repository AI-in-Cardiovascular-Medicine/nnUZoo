import random
from typing import Literal
from typing import Union, Tuple, List

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from batchgenerators.dataloading.data_loader import DataLoader
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from threadpoolctl import threadpool_limits


class CenterCropDataLoader(DataLoader):
    def __init__(self,
                 data: nnUNetBaseDataset,
                 batch_size: int,
                 initial_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager | None,
                 transforms=None,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 probabilistic_oversampling: bool = False,
                 target_type: Literal["ssl", "translation", "segmentation"] = "translation"):
        super().__init__(data, batch_size, 1, None, True, False, True, sampling_probabilities)

        assert isinstance(data, nnUNetBaseDataset), "CenterCropDataLoader only supports nnUNetBaseDataset."

        if len(initial_patch_size) == 2:
            final_patch_size = (1, *final_patch_size)
            initial_patch_size = (1, *initial_patch_size)
            self.patch_size_was_2d = True
        else:
            self.patch_size_was_2d = False

        self.indices = data.identifiers
        self.target_type = target_type
        self._data: nnUNetBaseDataset  # Set in the super class
        self.final_patch_size = final_patch_size
        self.patch_size = final_patch_size
        # self.list_of_keys = list(self._data.keys())
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the images
        self.need_to_pad = (np.array(initial_patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if self.patch_size_was_2d:
                pad_sides = (0, *pad_sides)
            for d in range(len(self.need_to_pad)):
                self.need_to_pad[d] += pad_sides[d]
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.sampling_probabilities = sampling_probabilities
        self.transforms = transforms

    def determine_shapes(self):
        # load one case
        data, seg, seg_prev, properties, *rest = self._data.load_case(self._data.identifiers[0])
        num_color_channels = data.shape[0]

        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        if seg is not None:
            channels_seg = seg.shape[0]
            if seg_prev is not None:
                channels_seg += 1
            seg_shape = (self.batch_size, channels_seg, *self.patch_size)
        else:
            seg_shape = None
        return data_shape, seg_shape

    # def get_bbox(
    #         self,
    #         data_shape: np.ndarray,
    # ):
    #     """Originally used to do probabilistic oversampling of foreground patches. We don't have foreground here though, so"""
    #     # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
    #     # locations for the given slice
    #     need_to_pad = self.need_to_pad.copy()
    #     dim = len(data_shape)
    #
    #     for d in range(dim):
    #         # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
    #         # always
    #         if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
    #             need_to_pad[d] = self.patch_size[d] - data_shape[d]
    #
    #     # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
    #     # define what the upper and lower bound can be to then sample form them with np.random.randint
    #     lbs = [-need_to_pad[i] // 2 for i in range(dim)]
    #     ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]
    #
    #     # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
    #     # at least one of the foreground classes in the patch
    #     bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
    #     bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
    #
    #     return bbox_lbs, bbox_ubs

    def generate_train_batch(self, index: int = None, pick_random: bool = False):
        if isinstance(index, tuple):
            index, pick_random = index
        # if index is not None:
        #     offset = index * self.batch_size
        #     case_properties = []
        #     selected_keys = []
        #     data_all = []
        #
        #     for i in range(self.batch_size):
        #         # oversampling foreground will improve stability of model training, especially if many patches are empty
        #         # (Lung for example)
        #         try:
        #             selected_key = self.indices[offset + i]
        #         except IndexError:
        #             continue
        #         selected_keys.append(selected_key)
        #
        #         data, seg, seg_prev, properties = self._data.load_case(selected_key)
        #         case_properties.append(properties)
        #
        #         # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
        #         # self._data.load_case(i) (see nnUNetDataset.load_case)
        #         shape = data.shape[1:]
        #         dim = len(shape)
        #         bbox_lbs, bbox_ubs = self.get_bbox(shape, pick_random)  # Returns the Center Crop.
        #
        #         # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
        #         # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
        #         # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
        #         # later
        #         # valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
        #         # valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]
        #
        #         # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
        #         # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
        #         # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
        #         # remove label -1 in the data augmentation but this way it is less error prone)
        #         # this_slice = tuple(
        #         #     [slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(bbox_lbs, bbox_ubs)]
        #         # )
        #         # data = data[this_slice]
        #         bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]
        #         data_all[j] = crop_and_pad_nd(data, bbox, 0)
        #         padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
        #         data_all.append(np.pad(data, ((0, 0), *padding), "constant", constant_values=0))
        #     data_all = np.stack(data_all, axis=0)
        #
        #     if self.patch_size_was_2d:
        #         data_all = data_all[:, :, 0]
        #     if self.target_type  == "ssl":
        #         return {"data": data_all, "properties": case_properties, "keys": selected_keys}
        #     else:
        #         return {"data": data_all, "target": seg_all,  "properties": case_properties, "keys": selected_keys}
        # else:
        if index is not None:
            selected_keys = list(range(self.batch_size))
            offset = index * self.batch_size
        else:
            selected_keys = self.get_indices()
            offset = None
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        if self.target_type == "translation":
            seg_all = np.zeros(self.seg_shape, dtype=np.float32)
        elif self.target_type == "segmentation":
            seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        elif self.target_type == "ssl":
            seg_all = None
        else:
            raise ValueError(f"target_type is not supported: {self.target_type}")
        case_properties = []
        j = 0
        for i in selected_keys.copy():
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            # force_fg = self.get_do_oversample(j)
            if isinstance(i, int):
                selected_key = self.indices[offset + i]
                selected_keys[i] = selected_key
                i = selected_key
            data, seg, seg_prev, properties, *rest = self._data.load_case(i)
            case_properties.append(properties)
            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]

            bbox_lbs, bbox_ubs = self.get_bbox(shape, pick_random)

            # bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(3)]

            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # use ACVL utils for that. Cleaner.
            data_all[j] = crop_and_pad_nd(data, bbox, 0)
            if seg_all is not None:
                if self.target_type == "segmentation":
                    seg_cropped = crop_and_pad_nd(seg, bbox, -1)
                elif self.target_type == "translation":
                    seg_cropped = crop_and_pad_nd(seg, bbox, 0)
                else:
                    raise ValueError(f"target_type is not supported: {self.target_type}")
                if seg_prev is not None:
                    seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None]))
                seg_all[j] = seg_cropped
            j += 1
        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            if seg_all is not None:
                seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    images = []
                    if self.target_type in ["ssl", "classification"]:
                        for b in range(self.batch_size):
                            tmp = self.transforms(**{'image': data_all[b]})
                            images.append(tmp['image'])
                        data_all = torch.stack(images)
                        del images
                    elif self.target_type in ['translation', 'segmentation']:
                        if self.target_type == "segmentation":
                            seg_all = torch.from_numpy(seg_all).to(torch.int16)
                        elif self.target_type == "translation":
                            seg_all = torch.from_numpy(seg_all).to(torch.float32)
                        else:
                            raise ValueError(f"target_type is not supported: {self.target_type}")
                        segs = []
                        for b in range(self.batch_size):
                            tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                            images.append(tmp['image'])
                            segs.append(tmp['segmentation'])
                        data_all = torch.stack(images)
                        if isinstance(segs[0], list):
                            seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                        else:
                            seg_all = torch.stack(segs)
                        del segs, images
                    else:
                        raise ValueError(f"target_type is not supported: {self.target_type}")
            # return {'data': data_all, 'target': seg_all, "properties": case_properties, 'keys': selected_keys}
        else:
            data_all = torch.tensor(data_all)
            if seg_all is not None:
                seg_all = torch.tensor(seg_all)
        return {'data': data_all, 'target': seg_all, "properties": case_properties, 'keys': selected_keys}

    def get_bbox(
            self,
            data_shape: np.ndarray,
            pick_random: bool = False
    ):
        """Originally used to do probabilistic oversampling of foreground patches. We don't have foreground here though, so"""
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        bbox_lbs = [int((lbs[i] + ubs[i]) / 2) for i in range(dim)]  # Always take the center (after padding)
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        if pick_random:
            chosen_length = bbox_ubs[0] - bbox_lbs[0]
            last_value = data_shape[0] - chosen_length - 1
            if last_value >= 1:
                z_lbs = random.randint(0, last_value)
                bbox_lbs[0] = z_lbs
                bbox_ubs[0] = z_lbs + chosen_length

        return bbox_lbs, bbox_ubs

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        return self.generate_train_batch(index)

