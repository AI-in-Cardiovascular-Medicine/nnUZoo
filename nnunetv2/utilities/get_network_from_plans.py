from typing import Literal

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

import inspect
import pydoc
import warnings
from typing import Union

from nnunetv2.utilities.find_class_by_name import recursive_find_python_class


def get_network_from_plans(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels,
                           allow_init=True, deep_supervision: Union[bool, None] = None,
                           up_sample_type: Literal["convtranspose", "trilinear", "nearest"] = "convtranspose",
                           add_skip_connection:bool=True, ssl_finetune_with_skip_connection:bool=False):
    network_class = arch_class_name
    architecture_kwargs = dict(**arch_kwargs)
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    nw_class = pydoc.locate(network_class)
    # sometimes things move around, this makes it so that we can at least recover some of that
    if nw_class is None:
        warnings.warn(f'Network class {network_class} not found. Attempting to locate it within '
                      f'nnunetv2.nets...')
        import nnunetv2.nets
        nw_class = recursive_find_python_class(nnunetv2.nets.__path__[0],
                                               network_class.split(".")[-1],
                                               'nnunetv2.nets')
        if nw_class is not None:
            print(f'FOUND IT: {nw_class}')
        else:
            raise ImportError('Network class could not be found, please check/correct your plans file')

    if deep_supervision is not None:
        architecture_kwargs['deep_supervision'] = deep_supervision

    # Only forward the nnUZoo-specific knobs to classes that actually declare them. Architectures in
    # nnunetv2/nets/ differ in which they support: the vendored U-Nets take add_skip_connection but not
    # up_sample_type, while several of the X2Net/Mamba nets are the other way round.
    #
    # A knob left at its default is dropped silently; a knob explicitly set to something else and not
    # accepted is an error. Silently ignoring it -- which is what the old bare `except: pop()` did -- means
    # "-up_sample_type trilinear" quietly trains a convtranspose model.
    try:
        accepted = set(inspect.signature(nw_class.__init__).parameters)
        takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD
                           for p in inspect.signature(nw_class.__init__).parameters.values())
    except (TypeError, ValueError):       # C-implemented or otherwise un-introspectable
        accepted, takes_kwargs = set(), True

    for kw_name, value, default in (('up_sample_type', up_sample_type, 'convtranspose'),
                                    ('add_skip_connection', add_skip_connection, True),
                                    ('ssl_finetune_with_skip_connection', ssl_finetune_with_skip_connection, False)):
        if kw_name in accepted or takes_kwargs:
            architecture_kwargs[kw_name] = value
        elif value != default:
            raise TypeError(
                f"{network_class} does not accept '{kw_name}', but it was set to {value!r} "
                f"(default {default!r}). Either pick an architecture that supports it or drop the option -- "
                f"silently ignoring it would train a different network than you asked for."
            )
    network = nw_class(
        input_channels=input_channels,
        num_classes=output_channels,
        **architecture_kwargs
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network

if __name__ == "__main__":
    import torch

    model = get_network_from_plans(
        arch_class_name="nnunetv2.nets.dynamic_network_architectures.unet.ResidualEncoderUNet",
        arch_kwargs={
            "n_stages": 7,
            "features_per_stage": [32, 64, 128, 256, 512, 512, 512],
            "conv_op": "torch.nn.modules.conv.Conv2d",
            "kernel_sizes": [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
            "strides": [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
            "n_blocks_per_stage": [1, 3, 4, 6, 6, 6, 6],
            "n_conv_per_stage_decoder": [1, 1, 1, 1, 1, 1],
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm2d",
            "norm_op_kwargs": {"eps": 1e-05, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        },
        arch_kwargs_req_import=["conv_op", "norm_op", "dropout_op", "nonlin"],
        input_channels=1,
        output_channels=4,
        allow_init=True,
        deep_supervision=True,
        add_skip_connection=False
    )
    data = torch.rand((8, 1, 256, 256))
    target = torch.rand(size=(8, 1, 256, 256))
    outputs = model(data) # this should be a list of torch.Tensor
    from torchinfo import summary
    summary(model, input_size=(1, 1, 256, 256))
    print("outputs: ", outputs.shape)
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    plans_manager = PlansManager("/home/aicvi/projects/nnUNet_translation/nnunetv2/nnUNetResEncUNetLPlans.json")
    configuration_manager = plans_manager.get_configuration("3d_fullres")
    # data = JsonUtils.load()
    # get_network_from_plans("nnunetv2.nets.dynamic_network_architectures.unet.ResidualEncoderUNet", data['configurations']['3d_fullres']["architecture"]["arch_kwargs"],
    #
    #                        )
    model = get_network_from_plans(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        1,
        2,

    )
    print(model)