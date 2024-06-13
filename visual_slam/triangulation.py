import numpy as np
from scipy.optimize import least_squares
import cv2


def triangulate_points(kp1, kp2, matches, pose1, pose2, mask, K):
    """Returns triangulated points and the index of corresponding keypoints and descriptors"""
    idx1 = [m.queryIdx for i, m in enumerate(matches) if mask[i]]
    idx2 = [m.trainIdx for i, m in enumerate(matches) if mask[i]]

    pts1 = np.float32([kp1[idx].pt for idx in idx1])
    pts2 = np.float32([kp2[idx].pt for idx in idx2])
   
    pts1_h = cv2.convertPointsToHomogeneous(pts1).squeeze().T
    pts2_h = cv2.convertPointsToHomogeneous(pts2).squeeze().T

    P1 = K @ pose1[:3]
    P2 = K @ pose2[:3]
   
    pts3d_h = triangulate(P1, P2, pts1_h[:2].T, pts2_h[:2].T)
    
    pts3d = cv2.convertPointsFromHomogeneous(pts3d_h).squeeze()


    # filter points
    reproj1 = P1 @ pts3d_h.T
    reproj2 = P2 @ pts3d_h.T
    visible_mask = (reproj1[2] > 0) & (reproj2[2] > 0)  # pts must be in front of camera
    
    pts3d = pts3d[visible_mask]
    idx1 = [idx for i, idx in enumerate(idx1) if visible_mask[i]]
    idx2 = [idx for i, idx in enumerate(idx2) if visible_mask[i]]

    return pts3d, idx1, idx2




def triangulate_points_lls(kp1, kp2, matches, pose1, pose2, mask, K):
    """Returns triangulated points and the index of corresponding keypoints and descriptors"""
    idx1 = [m.queryIdx for i, m in enumerate(matches) if mask[i]]
    idx2 = [m.trainIdx for i, m in enumerate(matches) if mask[i]]

    pts1 = np.float32([kp1[idx].pt for idx in idx1])
    pts2 = np.float32([kp2[idx].pt for idx in idx2])
   
    pts1_h = cv2.convertPointsToHomogeneous(pts1).squeeze().T
    pts2_h = cv2.convertPointsToHomogeneous(pts2).squeeze().T

    P1 = K @ pose1[:3]
    P2 = K @ pose2[:3]
   
    pts3d, reproj_err = iterative_linear_triangulation(P1, P2, pts1_h[:2].T, pts2_h[:2].T)

    # filter points
    pts3d_h = cv2.convertPointsToHomogeneous(pts3d).squeeze().T
    reproj1 = P1 @ pts3d_h
    reproj2 = P2 @ pts3d_h
    visible_mask = (reproj1[2] > 0) & (reproj2[2] > 0)  # pts must be in front of camera
    visible_mask = visible_mask & (reproj_err < 2.0)    # pts must have low reprojection error
    #visible_mask = visible_mask & (reproj1[2] < 100)
    
    pts3d = pts3d[visible_mask]
    idx1 = [idx for i, idx in enumerate(idx1) if visible_mask[i]]
    idx2 = [idx for i, idx in enumerate(idx2) if visible_mask[i]]

    return pts3d, idx1, idx2


# chatgpt ftw

def linear_triangulation(P1, P2, point1, point2):
    ''' Function to triangulate a point '''
    A = np.zeros((4, 4))
    A[0] = point1[0] * P1[2] - P1[0]
    A[1] = point1[1] * P1[2] - P1[1]
    A[2] = point2[0] * P2[2] - P2[0]
    A[3] = point2[1] * P2[2] - P2[1]
    
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def triangulate(pose1, pose2, pts1, pts2):
  ret = np.zeros((pts1.shape[0], 4))
  for i, p in enumerate(zip(pts1, pts2)):
    A = np.zeros((4,4))
    A[0] = p[0][0] * pose1[2] - pose1[0]
    A[1] = p[0][1] * pose1[2] - pose1[1]
    A[2] = p[1][0] * pose2[2] - pose2[0]
    A[3] = p[1][1] * pose2[2] - pose2[1]
    _, _, vt = np.linalg.svd(A)
    ret[i] = vt[3]
  return ret


def reprojection_error(X, P1, P2, x1, x2):
    """
    Compute the reprojection error for a single 3D point.
    """
    X_hom = np.hstack((X, 1))
    
    # Project points
    x1_proj = P1 @ X_hom
    x2_proj = P2 @ X_hom
    
    # Normalize homogeneous coordinates
    x1_proj = x1_proj[:2] / x1_proj[2]
    x2_proj = x2_proj[:2] / x2_proj[2]
    
    # Compute reprojection error
    error1 = x1 - x1_proj
    error2 = x2 - x2_proj
    
    return np.hstack((error1, error2))


def iterative_linear_triangulation(P1, P2, pts1, pts2):
    """
    Perform iterative linear triangulation to compute 3D points.
    
    Parameters:
    P1 : np.ndarray
        3x4 projection matrix for the first camera.
    P2 : np.ndarray
        3x4 projection matrix for the second camera.
    pts1 : np.ndarray
        Nx2 array of points in the first image.
    pts2 : np.ndarray
        Nx2 array of points in the second image.
    
    Returns:
    np.ndarray
        Nx3 array of triangulated 3D points.
    """
    num_points = pts1.shape[0]
    points_3D = np.zeros((num_points, 3))
    reproj_err = np.zeros((num_points))
    
    for i in range(num_points):
        x1 = pts1[i]
        x2 = pts2[i]
        
        # Initial guess using DLT
        X_initial = linear_triangulation(P1, P2, x1, x2)
        
        # Refine using least squares
        result = least_squares(
            reprojection_error, X_initial, args=(P1, P2, x1, x2)
        )
        
        points_3D[i] = result.x
        reproj_err[i] = result.cost
        
    
    return points_3D, reproj_err



