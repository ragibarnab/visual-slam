import numpy as np
import cv2
import glob
from structs import Frame, MapPoint, SLAMMap
from renderer import Renderer

calib_file = '/home/rae384/data/kitti/odometry/sequences/00/calib.txt'
with open(calib_file, 'r') as f:
    proj_mat = f.readline()
    proj_mat = proj_mat.strip().split(' ')[1:]
    proj_mat = np.float32(proj_mat).reshape(3, 4)
K = cv2.decomposeProjectionMatrix(proj_mat)[0]

img_seq_dir = '/home/rae384/data/kitti/odometry/sequences/00/image_0/'
img_seq = sorted(glob.glob(img_seq_dir + '*.png'))

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

slam_map = None
slam_map = None
initialized = False
frame_skip = 3
renderer = Renderer()

for frame_id, img_file in enumerate(img_seq):
    frame = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

    if not initialized:    # not initalized yet
        if slam_map is None:
            kp, des = orb.detectAndCompute(frame, None)
            slam_map = SLAMMap()
            slam_map.add_frame(np.eye(4), kp, des)
        elif frame_id == frame_skip + 1:
            prev_frame = slam_map.frames[-1]
            kp1, des1 = prev_frame.kp, prev_frame.des
            kp2, des2 = orb.detectAndCompute(frame, None)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
            E, mask = cv2.findEssentialMat(pts1, pts2, K)
            _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

            # triangulate points
            idx1 = [m.queryIdx for m in matches]
            idx2 = [m.trainIdx for m in matches]
            pts1 = np.float32([kp1[idx].pt for i, idx in enumerate(idx1) if mask[i]])
            pts2 = np.float32([kp2[idx].pt for i, idx in enumerate(idx2) if mask[i]])
            pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3).T
            pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1, 3).T
            P1 = K @ np.eye(4)[:3]
            P2 = K @ np.hstack([R, t])
            pts4d_h = cv2.triangulatePoints(P1, P2, pts1_h[:2], pts2_h[:2])
            pts3d = cv2.convertPointsFromHomogeneous(pts4d_h.T)
            pose = np.eye(4)
            pose[:3,:3] = R
            pose[:3, 3:] = t

            slam_map.add_frame(pose, kp2, des2)
            curr_frame = slam_map.frames[-1]

            # add observations and map points
            for i, pt in enumerate(pts3d):
                map_pt = MapPoint(pt)
                slam_map.add_map_point(map_pt)
                prev_frame.add_observation(map_pt, idx1[i])
                curr_frame.add_observation(map_pt, idx2[i])

            initialized = True

    else:

        # compute kp, des for current frame
        prev_frame = slam_map.frames[-1]
        kp1, des1 = orb.detectAndCompute(frame, None)
        kp2, des2 = prev_frame.kp, prev_frame.des

        # match with previous frame and perform pose estimation
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        E, mask = cv2.findEssentialMat(pts1, pts2, K)
        retval, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
        print(retval)
        rel_pose = np.eye(4)
        rel_pose[:3,:3] = R
        rel_pose[:3, 3:] = t
        curr_pose = prev_frame.pose @ rel_pose
        rvec, _ = cv2.Rodrigues(curr_pose[:3, :3])
        tvec = curr_pose[:3, 3:]
        
        pose = np.eye(4)
        rmat,_ = cv2.Rodrigues(rvec)
        pose[:3,:3] = rmat
        pose[:3, 3:] = tvec
        assert np.allclose(pose, curr_pose)

        # search for 3d to 2d correspondences between map points and current frame key points
        map_pts, proj_des = slam_map.get_visible_map_points_from_prev_frames()
        pts3d = np.float32([mp.pt3d for mp in map_pts])
        proj_pts, _ = cv2.projectPoints(pts3d, rvec, tvec, K, None)
        proj_pts = proj_pts.squeeze()

        matches = bf.match(proj_des, des2)

        for m in matches:
            if m.distance < 64:
                proj_pt = proj_pts[m.queryIdx]
                expected_pt = kp2[m.trainIdx].pt
                print(proj_pt, expected_pt)
                pass 
        #proj_pts = cv2.projectPoints()
        


        # triangulate matched points that does not have corresponding 3d map point

        # 
        exit()


    cv2.imshow('image display', frame)
    cv2.waitKey(100)

cv2.destroyAllWindows()