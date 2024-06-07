import cv2 
import numpy as np

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def get_kitti_calib(calib_file):
    with open(calib_file, 'r') as f:
        proj_mat = f.readline()
        proj_mat = proj_mat.strip().split(' ')[1:]
        proj_mat = np.float32(proj_mat).reshape(3, 4)
    K = cv2.decomposeProjectionMatrix(proj_mat)[0]
    return K


def match_features(des1, des2, threshold=50):
    """ Matches features between two sets of descriptors with a distance threshold. """
    matches = bf.match(des1, des2)
    matches = [m for m in matches if m.distance < threshold]
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def estimate_pose(kp1, kp2, matches, K):
    """ Estimates the pose using matched keypoints and returns inliers mask. """
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    E, mask = cv2.findEssentialMat(pts1, pts2, K)
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

    # for some f***ing reason opencv returns the inverse??
    pose = np.eye(4)
    pose[:3,:3] = R
    pose[:3, 3:] = t
    pose = np.linalg.inv(pose)

    return pose, mask_pose


def triangulate_points(kp1, kp2, matches, pose, mask, K):
    """Returns triangulated points and the index of correspoding keypoints and descriptors"""
    idx1 = [m.queryIdx for m in matches]
    idx2 = [m.trainIdx for m in matches]

    pts1 = np.float32([kp1[idx].pt for i, idx in enumerate(idx1) if mask[i]])
    pts2 = np.float32([kp2[idx].pt for i, idx in enumerate(idx2) if mask[i]])
   
    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3).T
    pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1, 3).T
   
    P1 = K @ np.eye(4)[:3]
    P2 = K @ pose[:3]
   
    pts4d_h = cv2.triangulatePoints(P1, P2, pts1_h[:2], pts2_h[:2])
    pts3d = cv2.convertPointsFromHomogeneous(pts4d_h.T)
    pts3d = pts3d.squeeze()

    return pts3d, idx1, idx2

def project_points(pts3d, cam_pose, K):
    
    proj_mat = K @ cam_pose[:3]
    pts3d_h = np.hstack([pts3d, np.ones((pts3d.shape[0], 1))])
    proj_pts_h = proj_mat @ pts3d_h.T
    proj_pts_h[:2] /= proj_pts_h[2]
    proj_pts = proj_pts_h[:2].T
    return proj_pts