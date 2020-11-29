
import os
import argparse
import numpy as np
import torch
import torchvision

ap = argparse.ArgumentParser()

ap.add_argument("--datapath", type=str, help='link to data folder')
ap.add_argument("--model", type=str, default=stackhourglass)
ap.add_argument("--split_file", type=str, help='link to split file')
ap.add_argument("--maxdisp",type=int, default=192, help='maximum disparity')
ap.add_argument("--epochs", type=int, default=200, help='default epochs')
ap.add_argument("--loadmodel", type=str, help="link to pretraind model")
ap.add_argument("--savemodel", type=str, help="path to saved model")
ap.add_argument("--no-cuda", action='store_true', default=True, help="enable CUDA training")
ap.add_argument("--seed", type=int, default=1, help='for replication of results')
ap.add_argument("--btrain", type=int, default=4)
ap.add_argument('--lr_scale', type=int, default=200, metavar='S', help='random seed')

args = ap.parse_args()
