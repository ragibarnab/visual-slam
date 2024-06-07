import numpy as np
import cv2
import glob
from visual_slam.viewer import Viewer
from visual_slam import VisualSLAM
from visual_slam.utils import get_kitti_calib


# /home/rae384/data/kitti/odometry/sequences/00/image_0/
img_seq_dir = '/home/rae384/data/kitti/odometry/dataset/sequences/00/image_0/'
img_seq = sorted(glob.glob(img_seq_dir + '*.png'))
calib_file = '/home/rae384/data/kitti/odometry/dataset/sequences/00/calib.txt'
K = get_kitti_calib(calib_file)

viewer = Viewer()
visual_slam = VisualSLAM(K)

for frame_id, img_file in enumerate(img_seq):
    frame = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

    slam_map = visual_slam.process_frame(frame)
    viewer.update(slam_map)    

    cv2.imshow('image display', frame)
    cv2.waitKey(500)

cv2.destroyAllWindows()