import torch
import numpy as np
from einops import rearrange


def gen_random_mask(x, mask_ratio, patch_size):
    N = x.shape[0]
    L = (x.shape[2] // patch_size) ** 2  # Considers W and H are the same!
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.randn(N, L, device=x.device)

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # generate the binary mask: 0 is keep 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return mask


def upsample_mask(mask, scale):
    assert len(mask.shape) == 2
    p = int(mask.shape[1] ** .5)
    return mask.reshape(-1, p, p). \
        repeat_interleave(scale, axis=1). \
        repeat_interleave(scale, axis=2)


def window_masking(x: torch.Tensor,
                   mask_token: torch.nn.Module,
                   r: int = 4,
                   remove: bool = False,
                   mask_len_sparse: bool = False,
                   mask_ratio: float = 0.75,
                   input_shape: str = "B H W C"):
    """
    The new masking method, masking the adjacent r*r number of patches together

    Optional whether to remove the mask patch,
    if so, the return value returns one more sparse_restore for restoring the order to x

    Optionally, the returned mask index is sparse length or original length,
    which corresponds to the different size choices of the decoder when restoring the image

    x: [N, L, D]
    r: There are r*r patches in a window
    remove: Whether to remove the mask patch
    mask_len_sparse: Whether the returned mask length is a sparse short length
    """
    x = rearrange(x, f'{input_shape} -> B (H W) C')
    B, L, D = x.shape
    assert int(L ** 0.5 / r) == L ** 0.5 / r
    d = int(L ** 0.5 // r)

    noise = torch.rand(B, d ** 2, device=x.device)
    sparse_shuffle = torch.argsort(noise, dim=1)
    sparse_restore = torch.argsort(sparse_shuffle, dim=1)
    sparse_keep = sparse_shuffle[:, :int(d ** 2 * (1 - mask_ratio))]

    index_keep_part = torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 + sparse_keep % d * r
    index_keep = index_keep_part
    for i in range(r):
        for j in range(r):
            if i == 0 and j == 0:
                continue
            index_keep = torch.cat([index_keep, index_keep_part + int(L ** 0.5) * i + j], dim=1)

    index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
    index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int32)
    for i in range(B):
        index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
    index_mask = torch.tensor(index_mask, device=x.device)

    index_shuffle = torch.cat([index_keep, index_mask], dim=1)
    index_restore = torch.argsort(index_shuffle, dim=1)

    if mask_len_sparse:
        mask = torch.ones([B, d ** 2], device=x.device)
        mask[:, :sparse_keep.shape[-1]] = 0
        mask = torch.gather(mask, dim=1, index=sparse_restore)
    else:
        mask = torch.ones([B, L], device=x.device)
        mask[:, :index_keep.shape[-1]] = 0
        mask = torch.gather(mask, dim=1, index=index_restore)

    if remove:
        x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
        x_masked = rearrange(x_masked, f'B (H W) C -> {input_shape}', H=int(x_masked.shape[1] ** 0.5))
        return x_masked, mask, sparse_restore
    else:
        x_masked = torch.clone(x)
        for i in range(B):
            x_masked[i:i + 1, index_mask.cpu().numpy()[i, :], :] = mask_token
        x_masked = rearrange(x_masked, f'B (H W) C -> {input_shape}', H=int(x_masked.shape[1] ** 0.5))
        return x_masked, mask


def patchify(imgs, patch_size, in_chans):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], in_chans, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(imgs.shape[0], h * w, p ** 2 * in_chans)
    return x


def unpatchify(x, patch_size, in_chans):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = patch_size
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, in_chans))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(x.shape[0], in_chans, h * p, w * p)
    return imgs


def unpatchify_mask(mask, patch_size, in_chans):
    """
    x: (N, w/p * h/p * in_chans)
    imgs: (N, 3, H, W)
    """

    p, q = patch_size if isinstance(patch_size, (list, tuple)) else (patch_size, patch_size)

    h_ = w_ = int(mask.shape[1] ** .5)
    assert h_ * w_ == mask.shape[1]

    # mask = torch.ones(size=(x.shape[0], h_* w_ * in_chans, p, q))
    # mask = mask * (1 - x.repeat())
    mask = mask[:, :, None, None].repeat(1, 1, p, q)
    mask = rearrange(mask, "n (h_h w_w in_c) p q -> n in_c (p h_h) (q w_w)", p=p, q=q, w_w=w_, h_h=h_, in_c=in_chans)
    return mask
