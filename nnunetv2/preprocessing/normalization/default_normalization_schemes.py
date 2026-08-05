from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from deep_utils import StringUtils
from numpy import number


class ImageNormalization(ABC):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = None
    kwargs = None

    def __init__(self, use_mask_for_norm: bool = None, intensityproperties: dict = None,
                 target_dtype: Type[number] = np.float32):
        assert use_mask_for_norm is None or isinstance(use_mask_for_norm, bool)
        self.use_mask_for_norm = use_mask_for_norm
        assert isinstance(intensityproperties, dict)
        self.intensityproperties = intensityproperties
        self.target_dtype = target_dtype

    @abstractmethod
    def run(self, image: np.ndarray, seg: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Image and seg must have the same shape. Seg is not always used
        """
        pass


class ZScoreNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        image = image.astype(self.target_dtype)
        if self.use_mask_for_norm is not None and self.use_mask_for_norm and seg is None:
            StringUtils.print("use_mask_for_norm is set to True while mask is not available")

        if self.use_mask_for_norm is not None and self.use_mask_for_norm and seg is not None:
            # negative values in the segmentation encode the 'outside' region (think zero values around the brain as
            # in BraTS). We want to run the normalization only in the brain region, so we need to mask the image.
            # The default nnU-net sets use_mask_for_norm to True if cropping to the nonzero region substantially
            # reduced the image size.

            mask = seg >= 0
            mean = image[mask].mean()
            std = image[mask].std()
            image[mask] = (image[mask] - mean) / (max(std, 1e-8))
        else:
            mean = image.mean()
            std = image.std()
            image = (image - mean) / (max(std, 1e-8))
        return image


class ClippedZScoreNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True
    clip_min: float = -1024.0
    clip_max: float = 4096.0

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype)
        image = np.clip(image, self.clip_min, self.clip_max)
        if self.use_mask_for_norm is not None and self.use_mask_for_norm and seg is not None:
            mask = seg >= 0
            mean = image[mask].mean()
            std = image[mask].std()
            image[mask] = (image[mask] - mean) / max(std, 1e-8)
        else:
            mean = image.mean()
            std = image.std()
            image = (image - mean) / max(std, 1e-8)
        return image
    
    
    @staticmethod
    def reverse(img_path: str, prediction):
        from deep_utils import SITKUtils
        arr = SITKUtils.get_array(img_path)
        mean = arr.mean()
        std = arr.std()
        prediction = prediction * max(std, 1e-8) + mean
        return prediction


class ClippedZScoreNormalization_0_100(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True
    clip_min: float = 0
    clip_max: float = 100

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype)
        image = np.clip(image, self.clip_min, self.clip_max)
        if self.use_mask_for_norm is not None and self.use_mask_for_norm and seg is not None:
            mask = seg >= 0
            mean = image[mask].mean()
            std = image[mask].std()
            image[mask] = (image[mask] - mean) / max(std, 1e-8)
        else:
            mean = image.mean()
            std = image.std()
            image = (image - mean) / max(std, 1e-8)
        return image


class ClippedSqrtNormalization_0_40(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True

    clip_min: float = 0.0
    clip_max: float = 40.0

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype)

        # 1. CLIP — floor negative artifacts at 0, cap extreme hotspots at 40.
        #    Keeps all real uptake (SUV up to 40), removes the ±200 artifacts.
        image = np.clip(image, self.clip_min, self.clip_max)

        # 2. SQRT COMPRESSION — tame the ~40x dynamic range so hotspots don't
        #    dominate the loss. Safe: image >= 0 after the clip.
        image = np.sqrt(image)

        # 3. FIXED SCALE — divide by a corpus-fixed constant (sqrt(clip_max))
        #    to land in [0, 1]. NOT per-image stats — this keeps the
        #    intensity->value mapping consistent for downstream regression.
        scale = np.sqrt(self.clip_max, dtype=self.target_dtype)  # sqrt(40) ≈ 6.3246
        image = image / max(scale, 1e-8)

        # Optional: zero-out background outside the mask, mirroring nnUNet's
        # use_mask_for_norm behavior. After clip+sqrt, background (0) maps to 0
        # already, so this is mostly a no-op but kept for parity.
        if self.use_mask_for_norm is not None and self.use_mask_for_norm and seg is not None:
            mask = seg >= 0
            image[~mask] = 0.0

        return image

    
    @staticmethod
    def reverse(img_path: str, prediction):
        scale = np.sqrt(ClippedSqrtNormalization_0_40.clip_max, dtype=prediction.dtype)  # sqrt(40) ≈ 6.3246
        prediction = prediction * max(scale, 1e-8)
        return prediction

    
class IVUSNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        image = image.astype(self.target_dtype)
        mean = image.mean()
        std = image.std()
        image = (image - mean) / (max(std, 1e-8))
        return image


class PETMedian9Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True
    kwargs = {"image_val": 9}

    def run(self, image: np.ndarray, seg: np.ndarray = None, image_val: float = None, **kwargs) -> np.ndarray:
        """
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        image = image.astype(self.target_dtype)
        if self.use_mask_for_norm is not None and self.use_mask_for_norm:
            raise NotImplementedError("PETMedianNormalization")
            # negative values in the segmentation encode the 'outside' region (think zero values around the brain as
            # in BraTS). We want to run the normalization only in the brain region, so we need to mask the image.
            # The default nnU-net sets use_mask_for_norm to True if cropping to the nonzero region substantially
            # reduced the image size.
            mask = seg >= 0
            mean = image[mask].mean()
            std = image[mask].std()
            image[mask] = (image[mask] - mean) / (max(std, 1e-8))
        else:
            image /= image_val
        return image


class PET02Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True
    import monai
    normalizer = monai.transforms.ScaleIntensityRanged(keys=["input_image"], a_min=0,
                                                       a_max=2,
                                                       b_min=0,
                                                       b_max=1,
                                                       clip=False)

    def run(self, image: np.ndarray, seg: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        image = image.astype(self.target_dtype)
        if self.use_mask_for_norm is not None and self.use_mask_for_norm:
            raise NotImplementedError("PETMedianNormalization")
        else:
            image = self.normalizer({"input_image": image})['input_image']
        return image

class PET05Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True
    import monai
    normalizer = monai.transforms.ScaleIntensityRanged(keys=["input_image"], a_min=0,
                                                       a_max=5,
                                                       b_min=0,
                                                       b_max=1,
                                                       clip=False)

    def run(self, image: np.ndarray, seg: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        image = image.astype(self.target_dtype)
        if self.use_mask_for_norm is not None and self.use_mask_for_norm:
            raise NotImplementedError("PETMedianNormalization")
        else:
            image = self.normalizer({"input_image": image})['input_image']
        return image

class PETMedian3Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True
    kwargs = {"image_val": 3}

    def run(self, image: np.ndarray, seg: np.ndarray = None, image_val: float = None, **kwargs) -> np.ndarray:
        """
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        image = image.astype(self.target_dtype)
        if self.use_mask_for_norm is not None and self.use_mask_for_norm:
            raise NotImplementedError("PETMedianNormalization")
            # negative values in the segmentation encode the 'outside' region (think zero values around the brain as
            # in BraTS). We want to run the normalization only in the brain region, so we need to mask the image.
            # The default nnU-net sets use_mask_for_norm to True if cropping to the nonzero region substantially
            # reduced the image size.
            mask = seg >= 0
            mean = image[mask].mean()
            std = image[mask].std()
            image[mask] = (image[mask] - mean) / (max(std, 1e-8))
        else:
            image /= image_val
        return image


class CTNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"
        image = image.astype(self.target_dtype)
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['std']
        lower_bound = self.intensityproperties['percentile_00_5']
        upper_bound = self.intensityproperties['percentile_99_5']
        image = np.clip(image, lower_bound, upper_bound)
        image = (image - mean_intensity) / max(std_intensity, 1e-8)
        return image


class NoNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        return image.astype(self.target_dtype)


class RescaleTo01Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype)
        image = image - image.min()
        image = image / np.clip(image.max(), a_min=1e-8, a_max=None)
        return image


class RGBTo01Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert image.min() >= 0, "RGB images are uint 8, for whatever reason I found pixel values smaller than 0. " \
                                 "Your images do not seem to be RGB images"
        assert image.max() <= 255, "RGB images are uint 8, for whatever reason I found pixel values greater than 255" \
                                   ". Your images do not seem to be RGB images"
        image = image.astype(self.target_dtype)
        image = image / 255.
        return image
