import numpy as np
from .triangulation import linear_triangulation
import cv2


def estimate_pose(kp1, kp2, matches, K):
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = compute_essential_matrix(pts1, pts2, K)
    R, t = recover_pose(E, pts1, pts2, K, mask)

    return R, t, mask


def estimate_pose_cv2(kp1, kp2, matches, K):
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(pts1, pts2, K)
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask)

    return R, t, mask


def normalize_points(points):
    ''' Point normalization function '''

    mean = np.mean(points, axis=0)
    std_dev = np.std(points, axis=0)
    scale = np.sqrt(2) / std_dev.mean()
    T = np.array([[scale, 0, -scale * mean[0]],
                  [0, scale, -scale * mean[1]],
                  [0, 0, 1]])
    normalized_points = np.dot(T, np.column_stack((points, np.ones(points.shape[0]))).T).T
    return normalized_points, T


def compute_fundamental_matrix(points1, points2):
    ''' Fundamental matrix computation '''
    n = points1.shape[0]
    A = np.zeros((n, 9))
    for i in range(n):
        x1, y1, _ = points1[i]
        x2, y2, _ = points2[i]
        A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
    
    # Solve A * f = 0 using SVD
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    
    # Enforce rank 2 constraint on F
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), Vt))
    return F


def compute_fundamental_matrix_ransac(points1, points2, threshold=0.01, iterations=1000):
    ''' RANSAC for robust fundamental matrix '''
    best_inliers = []
    best_F = None

    for _ in range(iterations):
        sample_indices = np.random.choice(len(points1), 8, replace=False)
        F_candidate = compute_fundamental_matrix(points1[sample_indices], points2[sample_indices])
        
        inliers = []
        for i in range(len(points1)):
            pt1 = points1[i]
            pt2 = points2[i]
            error = np.abs(np.dot(pt2, np.dot(F_candidate, pt1)))
            if error < threshold:
                inliers.append(i)
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_F = F_candidate

    mask = np.zeros(shape=points1.shape[0], dtype=bool)
    mask[best_inliers] = True
    return best_F, mask


def compute_essential_matrix(pts1, pts2, K):
    """ Compute the essential matrix given corresponding points and camera intrinsics. """
    
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    F, mask = compute_fundamental_matrix_ransac(pts1_norm, pts2_norm)

    # Denormalize to obtain the fundamental matrix for the original points
    F = T2.T @ F @ T1

    E = K.T @ F @ K

    U, S, Vt = np.linalg.svd(E)
    S = [1, 1, 0]  # force the singular values to be [1, 1, 0]
    E = np.dot(U, np.dot(np.diag(S), Vt))

    return E, mask


def extract_camera_pose(E):
    '''Helper function to extract rotation and translation from essential matrix'''
    U, _, Vt = np.linalg.svd(E)
    
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1
    
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    
    return [R1, R2], [t, -t]


def recover_pose(E, points1, points2, K, mask):
    '''Main function to recover the correct pose'''

    R_set, t_set = extract_camera_pose(E)
    
    best_R = None
    best_t = None
    max_in_front = -1
    
    for R in R_set:
        for t in t_set:
            in_front_count = check_cheirality(R, t, points1, points2, K)
            if in_front_count > max_in_front:
                max_in_front = in_front_count
                best_R = R
                best_t = t
    
    return best_R, best_t


# Function to check cheirality (points in front of both cameras)
def check_cheirality(R, t, points1, points2, K):
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t.reshape(3, 1)))
    
    in_front_count = 0
    for i in range(points1.shape[0]):
        X = linear_triangulation(P1, P2, points1[i], points2[i])
        if X[2] > 0 and (R @ X + t)[2] > 0:
            in_front_count += 1
    
    return in_front_count


