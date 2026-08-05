# Third-party notices — `nnunetv2/nets/dynamic_network_architectures/`

## dynamic-network-architectures

- **Upstream:** https://github.com/MIC-DKFZ/dynamic-network-architectures
- **Version vendored:** 0.4.2
- **Author:** Fabian Isensee, Division of Medical Image Computing, German Cancer Research Center (DKFZ)
- **Licence:** Apache License 2.0 — full text in [`LICENCE-dynamic-network-architectures`](LICENCE-dynamic-network-architectures)

nnUZoo vendors the subset of this package that it actually uses, so that it owns its
architecture code outright instead of tracking an external fork. Not vendored, because nnUZoo
does not use them: `architectures/{resnet,vgg,primus}.py` and `building_blocks/{eva,patch_encode_decode}.py`
(the latter two backed the transformer SSL trainers, which are out of scope).

| nnUZoo file | assembled from upstream modules | modified? |
|---|---|---|
| [`network_blocks.py`](network_blocks.py) | `building_blocks/{helper,regularization,simple_conv_blocks,residual,plain_conv_encoder,residual_encoders,unet_decoder,unet_residual_decoder}.py`, `initialization/weight_init.py`, `architectures/abstract_arch.py` | **yes** |
| [`unet.py`](unet.py) | `architectures/unet.py` | **yes** |

### Modifications

Copyright (c) 2026 The nnUZoo authors. Licensed under Apache License 2.0.

Each modified file carries a `NOTICE:` block in its module docstring, as required by
section 4(b) of the licence. In summary:

- **`network_blocks.py`** — `UNetDecoder` gained an `add_skip_connection: bool = True`
  parameter. When `False`, the encoder skips are not concatenated onto the transposed-conv
  output, so each decoder stage takes `1 x` rather than `2 x input_features_skip` channels.
  This is required for MAE self-supervised pretraining: with the skips connected, the decoder
  can copy the unmasked signal straight through and the reconstruction objective is trivial.
- **`unet.py`** — `PlainConvUNet` and `ResidualEncoderUNet` accept `add_skip_connection` and
  forward it to the decoder. `ResidualUNet` accepts it and raises `NotImplementedError` on
  `False` rather than silently building a skip-connected network.

With `add_skip_connection=True` (the default), these classes are behaviourally identical to
upstream 0.4.2: verified to produce byte-identical `state_dict` keys and shapes, and identical
forward outputs given identical weights, for `PlainConvUNet`, `ResidualEncoderUNet` and
`ResidualUNet` at both `deep_supervision=True` and `False`.

### Deliberate divergence from the previous nnUZoo-Base fork

The former fork at `nnUZoo-Base/dynamic-network-architectures/` additionally changed
`UNetDecoder` to build `seg_layers` only when `deep_supervision=True` (plus the final stage).
Upstream deliberately always builds them so that a `deep_supervision=True` checkpoint can be
loaded into a `deep_supervision=False` model. That change is **not** reproduced here, because
it silently breaks checkpoint compatibility and is unrelated to self-supervised learning.

Consequence: a checkpoint trained by the old fork at `deep_supervision=False` holds a single
`decoder.seg_layers.0.*` (shape `[C, features_per_stage[0], 1, 1, 1]`), whereas this code
expects `decoder.seg_layers.0` … `.{n_stages-2}`, with the fork's tensor corresponding to
index `n_stages-2`. Encoder parameters are unaffected, so `--only_encoder` pretrained-weight
loading is unaffected.

The fork's `up_sample_type` (`UNetDecoder_Upsample_{Nearest,Trilinear}`) and
`ssl_finetune_with_skip_connection` are also not vendored; nothing in the tree set the latter
to `True`.
