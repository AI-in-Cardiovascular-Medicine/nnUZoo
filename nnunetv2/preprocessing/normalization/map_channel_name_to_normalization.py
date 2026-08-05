from typing import Type


from nnunetv2.preprocessing.normalization.default_normalization_schemes import (CTNormalization,
                                                                                NoNormalization,
                                                                                ZScoreNormalization,
                                                                                RescaleTo01Normalization,
                                                                                RGBTo01Normalization,
                                                                                ImageNormalization,
                                                                                PETMedian9Normalization,
                                                                                PETMedian3Normalization,
                                                                                PET02Normalization,
                                                                                PET05Normalization,
                                                                                IVUSNormalization,
                                                                                ClippedZScoreNormalization,
                                                                                ClippedSqrtNormalization_0_40
                                                                                )

channel_name_to_normalization_mapping = {
    'CT': CTNormalization,
    'noNorm': NoNormalization,
    'zscore': ZScoreNormalization,
    'rescale_to_0_1': RescaleTo01Normalization,
    'rgb_to_0_1': RGBTo01Normalization,
    "PETMedian9": PETMedian9Normalization,
    "PETMedian3": PETMedian3Normalization,
    "PET05Normalization": PET05Normalization,
    "PET02Normalization": PET02Normalization,
    "clipped_zscore": ClippedZScoreNormalization,
    "clipped_sqrt_0_40": ClippedSqrtNormalization_0_40,
    "IVUS": IVUSNormalization,
}


def get_normalization_scheme(channel_name: str) -> Type[ImageNormalization]:
    """
    If we find the channel_name in channel_name_to_normalization_mapping return the corresponding normalization. If it is
    not found, use the default (ZScoreNormalization)
    """
    if channel_name not in channel_name_to_normalization_mapping:
        raise ValueError(f"{channel_name=} is not supported, Choose zscore if you don't know what to do")
    else:
        norm_scheme = channel_name_to_normalization_mapping[channel_name]
    return norm_scheme
