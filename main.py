import numpy as np
import cv2
import glob
from visual_slam.viewer import Viewer
from visual_slam import VisualSLAM
from visual_slam.utils import get_kitti_calib
from visual_slam.optimize import window_BA


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
    frame = cv2.drawKeypoints(frame, slam_map.get_last_frame().kp, None, color=(0,255,0), flags=0)

    cv2.imshow('image display', frame)
    cv2.waitKey(1)
    if frame_id == 200:
        #print(K[0,0], (K[0, 2], K[1, 2]))
        window_BA(slam_map, 1000, K)
        viewer.update(slam_map)
        cv2.waitKey(0)
        break

cv2.destroyAllWindows()