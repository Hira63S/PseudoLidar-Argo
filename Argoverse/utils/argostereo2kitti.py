"""Converting the Argoverse Data to KITTI-style dataset"""

import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
import os
from shutil import copyfile
from argoverse.utils import calibration
import json
import numpy as np
from argoverse.utils.calibration import CameraConfig
from argoverse.utils.cv2_plotting_utils import draw_clipped_line_segment
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat
import math
import os
from typing import Union
import numpy as np
import pyntcloud
import progressbar
from time import sleep

root_dir = 'C://users/cathx/repos/argoverse-api/'

max_d = 100

_PathLike = Union[str, 'os.PathLike[str]']

def load_ply(ply_fpath: _PathLike) -> np.ndarray:
    """
    Load a point cloud file from a filepath.
    Args:
        ply_fpath: Path to a PLY file
    Returns:
        arr: array of shape (N, 3)
    """
    data = pyntcloud.PyntCloud.from_file(os.fspath(ply_fpath))
    x = np.array(data.points.x)[:, np.newaxis]
    y = np.array(data.points.y)[:, np.newaxis]
    z = np.array(data.points.z)[:, np.newaxis]

    return np.concatenate((x, y, z), axis=1)

data_dir = root_dir + 'train/'
goal_dir = root_dir + 'train_test_kitti/'

if not os.path.exists(goal_dir):
    os.mkdir(goal_dir)
    os.mkdir(goal_dir+'velodyne')
    os.mkdir(goal_dir+'image_2')
    os.mkdir(goal_dir+'image_3')
    os.mkdir(goal_dir+'calib')
    os.mkdir(goal_dir+'label_2')
    os.mkdir(goal_dir+'velodyne_reduced')

argoverse_loader = ArgoverseTrackingLoader(data_dir)
print('\nTotal number of logs:',len(argoverse_loader))
argoverse_loader.print_all()
print('\n')

cams = ['stereo_front_left',
        'stereo_front_right']

total_number = 0
for q in argoverse_loader.log_list:
    path, dirs, files = next(os.walk(data_dir+q+'lidar'))
    total_number = total_number + len(files)

total_number = total_number*7

bar = progressbar.ProgressBar(maxval=total_number,
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

print('Total number of files: {}. Translation starts...'.format(total_number))
print('Progress:')
bar.start()

i = 0
for log_id in argoverse_loader.log_list:
    argoverse_data = argoverse_loader.get(log_id)
    for cam in cams:
        calibration_data = 
