# """
# This module contains augmentations that try to mimic different ct-scan devices.
# """
# import os
# import random
# from typing import List, Tuple, Union, Literal
#
# import numpy as np
# import torch
# from batchgenerators.transforms.abstract_transforms import AbstractTransform
# from deep_utils import DirUtils, JsonUtils
# from monai.transforms import Resize
# from monai.utils import InterpolateMode
#
#
# def monai_resize(img: np.ndarray, target_size, input_type: str):
#     input_dtype = img.dtype
#     if input_type == "image":
#         mode = InterpolateMode.BILINEAR
#     elif input_type == "label":
#         mode = InterpolateMode.NEAREST_EXACT
#     else:
#         raise ValueError(f"input_type: {input_type} is not correct!")
#     transform = Resize(spatial_size=target_size, mode=mode)
#     out_array = transform(img=torch.tensor(img[None, ...])).numpy()[0]
#     out_array = out_array.astype(input_dtype)
#     return out_array
#
#
# class StepAndShootAug:
#     def __init__(self, cutoff_index: int = -1,
#                  cut_off_pixel_value_weight: Union[Tuple[float, float], List[float]] = (0.2, 0.6),
#                  normalize: bool = True):
#         """
#         :param cutoff_index:
#         :param cut_off_pixel_value_weight: The mean pixel value is calculated then a value between the
#         given input values is randomly chosen and added to a part of the image.
#         :param normalize: whether to normalize the output between min and max. Default is True,
#         """
#
#         self.cutoff_index = cutoff_index
#         assert 0 <= cut_off_pixel_value_weight[0]
#         self.cut_off_pixel_value_weight = cut_off_pixel_value_weight
#         self.normalize = normalize
#
#     def __call__(self, data):
#         image: np.ndarray = data.copy().astype(np.float32)
#         if self.normalize:
#             min_value = np.min(image)
#             max_value = np.max(image)
#
#         cut_off_position = random.randint(0, image.shape[self.cutoff_index] - 1)
#         mean_pixel_value = np.mean(image)
#         intensity_value = random.uniform(self.cut_off_pixel_value_weight[0] * mean_pixel_value,
#                                          self.cut_off_pixel_value_weight[1] * mean_pixel_value)
#         intensity_value = intensity_value if random.random() < 0.5 else -intensity_value
#         # whether to lower the pixel values or increase it!
#
#         image = step_and_shoot_core(img=image,
#                                     cutoff_index=self.cutoff_index,
#                                     cut_off_position=cut_off_position,
#                                     intensity_value=intensity_value)
#         if self.normalize:
#             image = (image - np.min(image)) / (np.max(image) - np.min(image))  # normalize to 0-1
#             image = image * (max_value - min_value) + min_value  # scale to the requested range
#
#         return image
#
#
# def step_and_shoot_core(img: np.ndarray, cutoff_index: int, cut_off_position: int, intensity_value: float):
#     """
#     Core function for doing step and shoot augmentation
#     :param img:
#     :param cutoff_index:
#     :param cut_off_position:
#     :param intensity_value:
#     :return:
#     """
#     image = img.copy()
#     if random.random() < 0.5:
#         if cutoff_index == 0:
#             image[cut_off_position:, ...] += intensity_value
#         elif cutoff_index == 1:
#             image[:, cut_off_position:, ...] += intensity_value
#         elif cutoff_index == 2 or cutoff_index == -1:
#             image[:, :, cut_off_position:] += intensity_value
#         else:
#             raise ValueError("Only 3D images are supported!")
#     else:
#         if cutoff_index == 0:
#             image[:cut_off_position, ...] += intensity_value
#         elif cutoff_index == 1:
#             image[:, :cut_off_position, ...] += intensity_value
#         elif cutoff_index == 2 or cutoff_index == -1:
#             image[:, :, :cut_off_position] += intensity_value
#         else:
#             raise ValueError("Only 3D images are supported!")
#     return image
#
#
# class MotionAugmentation:
#     def __init__(self,
#                  motion_move_position: Union[Tuple[float, float], List[float]] = (0.05, 0.2),
#                  motion_move_range: Union[Tuple[float, float], List[float]] = (0.01, 0.05),
#                  normalize: bool = True,
#                  ):
#         """
#         :param motion_move_range: The mean pixel value is calculated then a value between the
#         given input values is randomly chosen and added to a part of the image.
#         :param normalize: whether to normalize the output between min and max. Default is True,
#         cut and resize the image to its original size, default is set to cut_resize!
#         """
#         # self.motion_index = motion_index
#         assert 0 <= motion_move_range[0]
#         assert 0 <= motion_move_position[0]
#         self.motion_move_range = motion_move_range
#         self.motion_move_position = motion_move_position
#         self.normalize = normalize
#
#     def __call__(self, data, seg_data, cropped_value: Literal["zero", "mean", "cut", "cut_resize"] = None,
#                  cutoff_index: int = None, move_index: int = None):
#         cropped_value = cropped_value or random.choice(["zero", "mean", "cut_resize"])
#         cutoff_index = cutoff_index or random.choice([1, 2])
#         move_index = move_index or random.choice(list({0, 1, 2} - {cutoff_index}))
#
#         image: np.ndarray = data.copy().astype(np.float32)
#         seg: np.ndarray = seg_data.copy().astype(np.float32)
#
#         if self.normalize:
#             image_min_value = np.min(image)
#             image_max_value = np.max(image)
#
#         image_mean_value = float(np.mean(image))
#         seg_mean_value = 0
#         img_size = image.shape
#
#         start = round(image.shape[move_index] * self.motion_move_range[0])
#         end = round(image.shape[move_index] * self.motion_move_range[1]) - 1
#         end = (start + 1) if end <= start else end
#         move_value = random.randint(start, end)
#
#         start = round(image.shape[cutoff_index] * self.motion_move_position[0])
#         end = round(image.shape[cutoff_index] * self.motion_move_position[1])
#         end = (start + 1) if end <= start else end
#         cut_off_position = random.randint(start, end)
#
#         cut_off_position = cut_off_position if random.random() < 0.5 else (image.shape[cutoff_index] - cut_off_position)
#         move_left = random.random() < 0.5
#
#         image = motion_core(image, cutoff_index, move_index, cropped_value,
#                             cut_off_position, img_size, image_mean_value, move_value,
#                             move_left=move_left, input_type="image")
#
#         seg = motion_core(seg, cutoff_index, move_index, cropped_value,
#                           cut_off_position, img_size, seg_mean_value, move_value,
#                           move_left=move_left, input_type="label")
#
#         if self.normalize:
#             image = (image - np.min(image)) / (np.max(image) - np.min(image))  # normalize to 0-1
#             image = image * (image_max_value - image_min_value) + image_min_value  # scale to the requested range
#         # print(f"move_index: {move_index}, cutoff_index: {cutoff_index},"
#         #       f" cropped_value: {cropped_value}, cut_off_position: {cut_off_position}, img: {image.shape}, seg: {seg.shape}")
#         return image, seg
#
#
# def motion_core(
#         image: np.ndarray,
#         cutoff_index: int,
#         move_index: int,
#         cropped_value: str,
#         cut_off_position: int,
#         img_size: tuple[int, ...],
#         mean_value: float,
#         move_value: int | float,
#         move_left: bool = None,
#         input_type: str = "image"):
#     image = image.copy()
#     move_left = move_left if move_left is not None else random.random() < 0.5
#
#     val = 0 if cropped_value == "zero" else mean_value
#     if cutoff_index == -1 or cutoff_index == 2:
#         if move_index == 1:
#             if move_left:
#                 image[:, :image.shape[1] - move_value, cut_off_position:] = image[:, move_value:,
#                                                                             cut_off_position:]
#             else:
#                 image[:, move_value:, cut_off_position:] = image[:, :image.shape[1] - move_value,
#                                                            cut_off_position:]
#
#             if cropped_value in ["zero", "mean"]:
#                 if move_left:
#                     image[:, image.shape[1] - move_value:, cut_off_position:] = val
#                 else:
#                     image[:, :move_value, cut_off_position:] = val
#             else:
#                 if move_left:
#                     image = image[:, :image.shape[1] - move_value, :]
#                 else:
#                     image = image[:, :move_value, :]
#         elif move_index == 0:
#             if move_left:
#                 image[:image.shape[0] - move_value, :, cut_off_position:] = image[move_value:, :,
#                                                                             cut_off_position:]
#             else:
#                 image[move_value:, :, cut_off_position:] = image[:image.shape[0] - move_value, :,
#                                                            cut_off_position:]
#             if cropped_value in ["zero", "mean"]:
#                 if move_left:
#                     image[image.shape[0] - move_value:, :, cut_off_position:] = val
#                 else:
#                     image[:move_value, :, cut_off_position:] = val
#             else:
#                 if move_left:
#                     image = image[:image.shape[0] - move_value, :, :]
#                 else:
#                     image = image[move_value:, :, :]
#         else:
#             raise ValueError(f"move_index: {move_index} is not valid!")
#
#     elif cutoff_index == 1:
#         if move_index == 0:
#             if move_left:
#                 image[:image.shape[0] - move_value, cut_off_position:, :] = image[move_value:,
#                                                                             cut_off_position:,
#                                                                             :]
#             else:
#                 image[move_value:, cut_off_position:, :] = image[:image.shape[0] - move_value,
#                                                            cut_off_position:, :]
#             if cropped_value in ["zero", "mean"]:
#                 if move_left:
#                     image[image.shape[0] - move_value:, cut_off_position:, :] = val
#                 else:
#                     image[:move_value, cut_off_position:, :] = val
#             else:
#                 if move_left:
#                     image = image[:image.shape[0] - move_value, :, :]
#                 else:
#                     image = image[move_value:, :, :]
#
#         elif move_index == 2:  # move_index == 2
#             if move_left:
#                 image[:, cut_off_position:, move_value:] = image[:, cut_off_position:,
#                                                            :image.shape[2] - move_value]
#             if cropped_value in ["zero", "mean"]:
#                 if move_left:
#                     image[:, cut_off_position:, image.shape[2] - move_value:] = val
#                 else:
#                     image[:, cut_off_position:, :move_value] = val
#             else:
#                 if move_left:
#                     image = image[:, :, :image.shape[2] - move_value]
#                 else:
#                     image = image[:, :, move_value:]
#         else:
#             raise ValueError(f"move_index: {move_index} is not valid!")
#     elif cutoff_index == 0:  # cutoff_index == 0:
#         if move_index == 1:
#             if move_left:
#                 image[cut_off_position:, :image.shape[1] - move_value, :] = image[cut_off_position:,
#                                                                             move_value:, :]
#             else:
#                 image[cut_off_position:, move_value:, :] = image[cut_off_position:,
#                                                            :image.shape[1] - move_value, :]
#             if cropped_value in ["zero", "mean"]:
#                 if move_left:
#                     image[cut_off_position:, image.shape[1] - move_value:, :] = val
#                 else:
#                     image[cut_off_position:, :move_value, :] = val
#             else:
#                 if move_left:
#                     image = image[:, :image.shape[1] - move_value, :]
#                 else:
#                     image = image[:, move_value:, :]
#         elif move_index == 2:  # move_index == 2
#             if move_left:
#                 image[cut_off_position:, :, :image.shape[2] - move_value] = image[cut_off_position:, :,
#                                                                             move_value:]
#             else:
#                 image[cut_off_position:, :, move_value:] = image[cut_off_position:, :,
#                                                            :image.shape[2] - move_value]
#             if cropped_value in ["zero", "mean"]:
#                 if move_left:
#                     image[cut_off_position:, :, image.shape[2] - move_value:] = val
#                 else:
#                     image[cut_off_position:, :, :move_value] = val
#             else:
#                 if move_left:
#                     image = image[:, :, :image.shape[2] - move_value]
#                 else:
#                     image = image[:, :, move_value:]
#         else:
#             raise ValueError(f"move_index: {move_index} is not valid!")
#     else:
#         raise ValueError(f"cutoff_index: {cutoff_index} is not valid!")
#
#     if cropped_value == "cut_resize":
#         image = monai_resize(image, img_size, input_type=input_type)
#     return image
#
#
# class StepAndShootAugPlusMotionAugmentation:
#     def __init__(self,
#                  cut_off_pixel_value_weight: tuple[float, float] | list[float] = (0.2, 0.6),
#                  motion_move_range: Union[Tuple[float, float], List[float]] = (0.02, 0.1),
#                  normalize: bool = True):
#         """
#         :param cut_off_pixel_value_weight: The mean pixel value is calculated then a value between the
#         given input values is randomly chosen and added to a part of the image.
#         :param normalize: whether to normalize the output between min and max. Default is True,
#         """
#         assert 0 <= cut_off_pixel_value_weight[0]
#         assert 0 <= motion_move_range[0]
#         self.cut_off_pixel_value_weight = cut_off_pixel_value_weight
#         self.motion_move_range = motion_move_range
#         self.normalize = normalize
#
#     def __call__(self, data, seg_data, cropped_value: Literal["zero", "mean", "cut", "cut_resize"] = None,
#                  cutoff_index: int = None, move_index: int = None):
#         cropped_value = cropped_value or random.choice(["zero", "mean", "cut_resize"])
#         cutoff_index = cutoff_index or 0
#         move_index = move_index or random.choice(list({0, 1, 2} - {cutoff_index}))
#
#         image: np.ndarray = data.copy().astype(np.float32)
#         seg: np.ndarray = seg_data.copy()
#
#         if self.normalize:
#             image_min_value = np.min(image)
#             image_max_value = np.max(image)
#
#         image_mean_value = float(np.mean(image))
#         seg_mean_value = 0
#
#         cut_off_position = random.randint(0, image.shape[cutoff_index] - 1)
#
#         img_size = image.shape
#
#         try:
#             start = round(image.shape[move_index] * self.motion_move_range[0])
#             end = round(image.shape[move_index] * self.motion_move_range[1]) - 1
#             end = (start + 1) if end <= start else end
#             move_value = random.randint(start, end)
#         except Exception as e:
#             raise ValueError(f"Error in random.randint with image.shape: {image.shape} and {self.motion_move_range}")
#
#         intensity_value = random.uniform(self.cut_off_pixel_value_weight[0] * image_mean_value,
#                                          self.cut_off_pixel_value_weight[1] * image_mean_value)
#         intensity_value = intensity_value if random.random() < 0.5 else -intensity_value
#
#         # whether to lower the pixel values or increase it!
#
#         move_left = random.random() < 0.5
#         # print(
#         #     f"[INFO] CutOff Position: {cut_off_position}, move_value: {move_value}, intensity: {intensity_value}, "
#         #     f"move_left: {move_left}, cropped_value: {cropped_value}, move_index: {move_index}")
#         image = step_and_shoot_core(img=image,
#                                     cutoff_index=cutoff_index,
#                                     cut_off_position=cut_off_position,
#                                     intensity_value=intensity_value)
#
#         image = motion_core(image, cutoff_index, move_index, cropped_value,
#                             cut_off_position, img_size, image_mean_value, move_value,
#                             move_left=move_left,
#                             input_type="image")
#
#         seg = motion_core(seg, cutoff_index, move_index, cropped_value,
#                           cut_off_position, img_size, seg_mean_value, move_value,
#                           move_left=move_left,
#                           input_type="label")
#
#         if self.normalize:
#             image = (image - np.min(image)) / (np.max(image) - np.min(image))  # normalize to 0-1
#             image = image * (image_max_value - image_min_value) + image_min_value  # scale to the requested range
#         # print(f"move_index: {move_index}, cutoff_index: {cutoff_index},"
#         #       f" cropped_value: {cropped_value}, cut_off_position: {cut_off_position}, img: {image.shape}, seg: {seg.shape}")
#         info = dict(move_index=move_index, cutoff_index=cutoff_index, cropped_value=cropped_value,
#                     cut_off_position=cut_off_position, move_value=move_value, move_left=move_left)
#         return image, seg, info
#
#
# class MotionGeneration(AbstractTransform):
#     def __init__(self, key_origin="image", key_target="image", ):
#         self.key_target = key_target
#         self.key_origin = key_origin
#         self.augmentor = MotionAugmentation(motion_move_range=(0.01, 0.015))
#
#     def __call__(self, **data_dict):
#         data = data_dict[self.key_origin][0].numpy().copy()
#         seg = data_dict["segmentation"][0].numpy().copy()
#         initial_size = data.shape
#         data, seg = self.augmentor(data, seg, random.choice(["zero", "mean", "cut_resize"]))
#         if data.shape != initial_size:
#             raise ValueError(f"Error in MotionGeneration for size, {data.shape} and {initial_size}")
#         data_dict[self.key_target] = torch.tensor(data[None, ...])
#         data_dict["segmentation"] = torch.tensor(seg[None, ...])
#
#         return data_dict
#
#
# class StepAndShootGeneration(AbstractTransform):
#     def __init__(self, key_origin="image", key_target="image"):
#         self.key_target = key_target
#         self.key_origin = key_origin
#         self.augmentor = StepAndShootAug()
#
#     def __call__(self, **data_dict):
#         data = data_dict[self.key_origin][0].numpy().copy()
#         initial_size = data.shape
#         data = self.augmentor(data)
#         if data.shape != initial_size:
#             raise ValueError(f"Error in StepAndShootAugPlusMotionGeneration for size, {data.shape} and {initial_size}")
#         data_dict[self.key_target] = torch.tensor(data[None, ...])
#         return data_dict
#
#
# class StepAndShootAugPlusMotionGeneration(AbstractTransform):
#     def __init__(self, key_origin="image", key_target="image"):
#         self.key_target = key_target
#         self.key_origin = key_origin
#         self.augmentor = StepAndShootAugPlusMotionAugmentation(motion_move_range=(0.01, 0.025))
#
#     def __call__(self, **data_dict):
#         data = data_dict[self.key_origin][0].numpy().copy()
#         seg = data_dict["segmentation"][0].numpy().copy()
#         initial_size = data.shape
#
#         data, seg, info = self.augmentor(data, seg)
#
#         if data.shape != initial_size:
#             raise ValueError(f"Error in StepAndShootAugPlusMotionGeneration for size, {data.shape} and {initial_size}")
#
#         data_dict[self.key_target] = torch.tensor(data[None, ...])
#         data_dict["segmentation"] = torch.tensor(seg[None, ...])
#         data_dict['info'] = info
#         return data_dict
#
#
# def do_multi_testing(n, output_path, crop: bool = True, keep_name: bool = False):
#     from deep_utils import SITKUtils
#     import os
#
#     img_names = os.listdir(
#         "/media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_raw/Dataset066_china_zurich_bern_no_coronary/imagesTs")
#     for i in range(n):
#         name = random.choice(img_names)
#         img_arr, img_img = SITKUtils.get_array_img(
#             f"/media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_raw/Dataset066_china_zurich_bern_no_coronary/imagesTs/{name}")
#         seg_arr, seg_img = SITKUtils.get_array_img(
#             f"/media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_raw/Dataset066_china_zurich_bern_no_coronary/labelsTs/{name.replace("_0000", "")}")
#         if crop:
#
#             exact = [128, 256, 224]
#             z_s = random.randint(0, img_arr.shape[0] - exact[0])
#             z_e = z_s + exact[0]
#
#             x_s = random.randint(0, img_arr.shape[1] - exact[1])
#             x_e = x_s + exact[1]
#
#             y_s = random.randint(0, img_arr.shape[2] - exact[2])
#             y_e = y_s + exact[2]
#             print(img_arr.shape)
#             chosen_arr = img_arr[z_s: z_e, x_s: x_e, y_s:y_e]
#             chosen_seg = seg_arr[z_s: z_e, x_s: x_e, y_s:y_e]
#         # SITKUtils.save_sample("arr_sample.nii.gz", chosen_arr, img=img_img)
#         # SITKUtils.save_sample("seg_sample.nii.gz", chosen_seg, img=seg_img)
#             print(chosen_arr.shape, chosen_seg.shape)
#
#         # img_arr, img_img = SITKUtils.get_array_img("arr_sample.nii.gz")
#         # seg_arr, seg_img = SITKUtils.get_array_img("seg_sample.nii.gz")
#             img_arr = chosen_arr
#             seg_arr = chosen_seg
#
#         augmentor = StepAndShootAugPlusMotionGeneration()
#         augmented_array = augmentor(image=torch.tensor(img_arr[None, ...]),
#                                     segmentation=torch.tensor(seg_arr[None, ...]))
#         # print(augmented_array['image'].shape)
#         if keep_name:
#             os.makedirs(f"{output_path}/images", exist_ok=True)
#             os.makedirs(f"{output_path}/segs", exist_ok=True)
#             os.makedirs(f"{output_path}/infos", exist_ok=True)
#             SITKUtils.save_sample(f"{output_path}/images/{name}", augmented_array['image'].numpy()[0],
#                                   img=img_img)
#             SITKUtils.save_sample(f"{output_path}/segs/{name}",
#                                   augmented_array['segmentation'].numpy()[0],
#                                   img=seg_img)
#             JsonUtils.dump(f"{output_path}/infos/{name[:-7]}.json", augmented_array["info"])
#         else:
#             os.makedirs(f"{output_path}/{i}_{name[:-7]}")
#             SITKUtils.save_sample(f"{output_path}/{i}_{name[:-7]}/img.nii.gz", augmented_array['image'].numpy()[0], img=img_img)
#             SITKUtils.save_sample(f"{output_path}/{i}_{name[:-7]}/seg.nii.gz", augmented_array['segmentation'].numpy()[0],
#                                   img=seg_img)
#             JsonUtils.dump(f"{output_path}/{i}_{name[:-7]}/info.json", augmented_array["info"])
#
#
# if __name__ == '__main__':
#     from deep_utils import DirUtils
#     output_path = "output-2"
#     DirUtils.remove_create(output_path)
#     # os.makedirs("output", exist_ok=True)
#     do_multi_testing(20, output_path, crop=False, keep_name=True)
#     # from deep_utils import SITKUtils
#     #
#     # data_path = "/media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_raw/Dataset066_china_zurich_bern_no_coronary/imagesTr/04-90004_0000.nii.gz"
#     # seg_path = "/media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_raw/Dataset066_china_zurich_bern_no_coronary/labelsTr/04-90004.nii.gz"
#     #
#     # arr, arr_img = SITKUtils.get_array_img(data_path)
#     # seg, seg_img = SITKUtils.get_array_img(seg_path)
#     #
#     # print(arr.shape)
#     #
#     # aug = StepAndShootAugPlusMotionAugmentation(motion_move_range=(0.01, 0.02))
#     # out_arr, out_seg = aug(arr, seg)
#     #
#     # SITKUtils.save_sample("seg.nii.gz", out_seg, img=seg_img)
#     # SITKUtils.save_sample("arr.nii.gz", out_arr, img=arr_img)
# # 20_04-90357_0000
