"""Model Utils """

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math

def convbn(inplanes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stirde=stride, padding=dilation if dilation  >1 else pad, dilation=dilation, bias=False),
                        nn.BatchNorm2d(out_planes))

def convbn_3d(inplanes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
                        nn.BatchNorm2d(out_planes))
