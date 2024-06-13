import cv2 
import numpy as np


def Rt_to_pose(R, t):
    pose = np.eye(4)
    pose[:3,:3] = R
    pose[:3, 3:] = t
    return pose

def pose_to_Rt(pose):
    R = pose[:3,:3] 
    t = pose[:3, 3:] 
    return R, t


def project_points(pts3d, cam_pose, K):
    
    proj_mat = K @ cam_pose[:3]
    pts3d_h = np.hstack([pts3d, np.ones((pts3d.shape[0], 1))])
    proj_pts_h = proj_mat @ pts3d_h.T
    proj_pts_h[:2] /= proj_pts_h[2]
    proj_pts = proj_pts_h[:2].T
    return proj_pts

def get_kitti_calib(calib_file):
    with open(calib_file, 'r') as f:
        proj_mat = f.readline()
        proj_mat = proj_mat.strip().split(' ')[1:]
        proj_mat = np.float32(proj_mat).reshape(3, 4)
    K = cv2.decomposeProjectionMatrix(proj_mat)[0]
    return K