from typing import Union
import numpy as np
import torch


def mask_creation(
        batch_size: int, patch_size: Union[tuple[int, int], tuple[int, int, int]], mask_ratio: float = 0.75,
        rng_seed: int = None, block_size=16, return_patch_size: bool = True, channel_dimention: int = 1
) -> torch.Tensor:
    """
    Creates a masking tensor with 1s (indicating no masking) and 0s (indicating masking).
    The mask has to be of same size like the input data (batch_size, 1, x, y, z).

    :param batch_size: The 3D shape information for the masking patch.
    :param patch_size: The 3D shape information for the masking patch.
    :param mask_ratio: percentage of the patch that should be masked
    :param rng_seed: minimum size of the blocks that should be masked
    :param block_size: Size of the block!
    :return:
    """

    sparsity_factor = mask_ratio
    c_masks = []
    for c_item in range(channel_dimention):
        mask_ = [create_blocky_mask(patch_size, block_size, sparsity_factor, rng_seed) for _ in
                 range(batch_size)]
        mask_ = torch.stack(mask_)[:, None, ...]  # Add channel dimension
        c_masks.append(mask_)
    mask = torch.concat(c_masks, dim=1)
    # mask = mask.repeat_interleave(channel_dimention, dim=1)
    if return_patch_size:
        # Make the mask the same size as the data
        mask_shape = mask.shape
        rep = [p_s // m_s for p_s, m_s in zip(patch_size, mask_shape[2:])]

        for index, val in enumerate(rep):
            mask = mask.repeat_interleave(val, dim=2 + index) # 0 and 1 are batch size and dimention which +2 skips it!
        one_mask = torch.ones((batch_size, channel_dimention, *patch_size))
        if len(patch_size) == 2:
            one_mask[:, :, :mask.shape[2], :mask.shape[3]] = mask
            mask = one_mask
        else:
            one_mask[:, :, :mask.shape[2], :mask.shape[3], :mask.shape[4]] = mask
            mask = one_mask
    return mask


def create_blocky_mask(tensor_size, block_size, sparsity_factor=0.75, rng_seed: None | int = None) -> torch.Tensor:
    """
    Create the smallest binary mask for the encoder by choosing a percentage of pixels at that resolution..

    :param tensor_size: Tuple of the dimensions of the tensor (height, width, depth).
    :param block_size: Size of the block to be masked (set to 0) in the smaller mask.
    :return: A binary mask tensor.
    """
    # Calculate the size of the smaller mask

    small_mask_size = tuple(size // block_size for size in tensor_size)
    while 0 in small_mask_size:
        block_size = max(1, block_size // 2)
        small_mask_size = tuple(size // block_size for size in tensor_size)

    # Create the smaller mask
    flat_mask = torch.ones(np.prod(small_mask_size))
    n_masked = int(sparsity_factor * flat_mask.shape[0])
    if rng_seed is None:
        mask_indices = torch.randperm(flat_mask.shape[0])[:n_masked]
    else:
        print("[INFO] rng_seed:", rng_seed)
        gen = torch.Generator.manual_seed(rng_seed)
        mask_indices = torch.randperm(flat_mask.shape[0], generator=gen)[:n_masked]
    flat_mask[mask_indices] = 0
    small_mask = torch.reshape(flat_mask, small_mask_size)
    return small_mask


if __name__ == '__main__':
    mask = mask_creation(1, (212, 212, 212), channel_dimention=1, mask_ratio=0.5)
    print(mask.shape)
    mask = mask.numpy()
    print(mask.sum())
    import matplotlib.pyplot as plt

    plt.imshow(mask[0, 0])
    plt.show()
