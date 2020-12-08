from torch.autograd import Function
import torch.nn as nn
import torchvision
import numpy as np

class BaseConv(Function):

    def __init__(input_features):
