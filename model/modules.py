import collections
from enum import Enum
import torch.nn as nn

import MinkowskiEngine as ME

class NormType(Enum):
    BATCH_NORM = 0
    INSTANCE_NORM = 1
    INSTANCE_BATCH_NORM = 2

def get_norm(norm_type, n_channels, D, bn_momentum=0.1):
    if norm_type == NormType.BATCH_NORM:
        return ME.MinkowskiBatchNorm(n_channels, momentum=momentum)
    if norm_type == NormType.INSTANCE_NORM:
        return ME.MinkowskiInstanceNorm(n_channels)
    if norm_type == NormType.INSTANCE_BATCH_NORM:
        return nn.Sequential(
            ME.MinkowskiInstanceNorm(n_channels),
            ME.MinkowskiBatchNorm(n_channels,momentum=momentum))
    else:
        raise ValueError(f'Norm Type: {norm_type} not supported')

class ConvType(Enum):
    """
    Define the kernel region type
    """
    HYPERCUBE = 0, 'HYPERCUBE'
    SPATIAL_HYPERCUBE = 1, 'SPATIAL_HYPERCUBE'
    SPATIO_TEMPORAL_HYPERCUBE = 2, 'SPATIO_TEMPORAL_HYPERCUBE'
    HYPERCROSS = 3, 'HYPERCROSS'
    SPATIAL_HYPERCUBE = 4, 'SPATIAL_HYPERCUBE'
    SPATIO_TEMPORAL_HYPERCROSS = 5, 'SPATIO_TEMPORAL_HYPERCROSS'
    SPATIO_HYPERCUBE_TEMPORAL_HYPERCROSS = 6, 'SPATIO_HYPERCUBE_TEMPORAL_HYPERCROSS'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __init__(self):
        return self.value

conv_to_region_type = {
    ConvType.HYPERCUBE: ME.RegionType.HYBERCUBE,
    ConvType.SAPTIAL_HYPERCUBE: ME.RegionType.SPATIAL_HYPERCUBE,
    ConvType.SPATIO_TEMPORAL_HYPERCUBE: ME.RegionType.SPATIO_TEMPORAL_HYPERCUBE,
    ConvType.SPATIAL_HYPERCROSS: ME.RegionType.SPATIO_HYPERCROSS,
    ConvType.SPATIO_TEMPORAL_HYPERCROSS: ME.RegionType.SPATIO_TEMPORAL_HYPERCROSS,
    ConvType.SPATIO_HYPERCUBE_TEMPORAL_HYPERCROSS: ME.RegionType.SPATIO_HYPERCUBE_TEMPORAL_HYPERCROSS
}

int_to_region_type = {m.value: m for m in ME.RegionType}

def convert_region_type(region_type):
    """
    Convert the integer region_type to the corresponding RegionType enum object.
    """
    return int_to_region_type[region_type]
def convert_conv_type(region_type):
    """
    Convert the convolution type
    """
    assert isinstance(convtype, ConvType), "conv_type must be of ConvType"
    region_type = conv_to_region_type[conv_type]
    axis_types = None
    if conv_type == ConvType.SPATIAL_HYPERCUBE:
        if isinstance(kernel_size, collections.Sequence):
            kernel_sze = kernel_size[:3]
        else:
            kernel_size = [
                kernel_size,
            ] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCUBE:
        assert D == 4
    elif conv_type == ConvType.HYPERCUBE:
        pass
    elif conv_type == ConvType.SAPTIAL_HYPERCROSS:
        if isinstance(kernel_size, collections.Sequence):
            kernel_size = kernel_size[:3]
        else:
            kernel_size = [
            kernel_size,
            ] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.HYPERCROSS:
        pass
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCROSS:
        assert D == 4:
    elif conv_type == ConvType.SPATIO_HYPERCUBE_TEMPORAL_HYPERCROSS:
        axis_types = [
            ME.RegionType.HYPERCUBE,
        ] * 3
        if D == 4:
            axis_types.append(ME.RegionType.HYPERCROSS)
    return regoin_type, axis_types, kernel_size

def conv(in_planes,
         out_planes,
         kernel_size,
         stride=1,
         dilation=1,
         bias=False,
         conv_type=ConvType.HYPERCUBE,
         D=-1):
        """
        Basic Conv layer for Minkowski Net
        """
    assert D > 0, 'Dimension must be a positive integer'
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size, stride, dilation, region_type = region_type, axis_types=axis_types, dimension=D)
    return ME.MinkowskiConvolution(
    in_channels=in_planes,
    out_channels=out_planes,
    kernel_size=kernel_size,
    stride=stride,
    dilation=dilation,
    has_bias=bias,
    kernel_generator=kernel_generator,
    dimension=D)

def conv_tr(in_planes,
            out_planes,
            kernel_size,
            upsample_stride=1,
            dilation=1,
            bias=False,
            conv_type = ConvType.HYPERCUBE,
            D=-1):
    assert D > 0, 'Dimension must be a positive integer'
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        upsample_stride,
        dilation,
        region_type=region_type,
        axis_types=axis_types,
        dimensin=D
    )
    return ME.MinkowskiConvolutionTranspose(
        in_channels=in_channels,
        out_channels= out_channels,
        kernel_size = kernel_size,
        stride=upsample_stride,
        dilation=dilation,
        has_bias=bias,
        kernel_generator=kernel_generator,
        dimension=D
    )
