import numpy as np
import cv2
from .structs import SLAMMap, Frame, MapPoint
from .utils import orb
from .utils import match_features, estimate_pose, triangulate_points, project_points


class VisualSLAM():

    def __init__(self, K) -> None:
        self.K = K
        self.slam_map = None
        self.initialized = False

    def process_frame(self, frame) -> SLAMMap:

        if not self.initialized:
            if self.slam_map is None:
                kp, des = orb.detectAndCompute(frame, None)
                self.slam_map = SLAMMap()
                self.slam_map.add_frame(np.eye(4), kp, des)
            else:
                prev_frame = self.slam_map.frames[-1]

                kp1, des1 = prev_frame.kp, prev_frame.des
                kp2, des2 = orb.detectAndCompute(frame, None)
                
                matches = match_features(des1, des2)

                rel_pose, mask = estimate_pose(kp1, kp2, matches, self.K)

                pts3d, idx1, idx2 = triangulate_points(kp1, kp2, matches, rel_pose, mask, self.K)

                # add current frame to map
                self.slam_map.add_frame(rel_pose, kp2, des2)
                curr_frame = self.slam_map.frames[-1]

                # add observations and map points
                for i, pt in enumerate(pts3d):
                    map_pt = MapPoint(pt)
                    self.slam_map.add_map_point(map_pt)
                    prev_frame.add_observation(map_pt, idx1[i])
                    curr_frame.add_observation(map_pt, idx2[i])

                self.initialized = True
        else:
            # compute kp, des for current frame
            prev_frame = self.slam_map.frames[-1]
            kp1, des1 = prev_frame.kp, prev_frame.des
            kp2, des2 = orb.detectAndCompute(frame, None)

            matches = match_features(des1, des2)

            rel_pose, mask = estimate_pose(kp1, kp2, matches, self.K)
            curr_pose = prev_frame.pose @ rel_pose

            # project map points to current frame to find correspondences
            proj_map_pts, proj_des = self.slam_map.get_visible_map_points_from_prev_frames()
            pts3d = np.float32([mp.pt3d for mp in proj_map_pts])
            proj_pts = project_points(pts3d, curr_pose, self.K)

            # match projected map points to current frame's features
            proj_matches = match_features(proj_des, des2, threshold=50)
            obs = np.float32([kp2[m.trainIdx].pt for m in proj_matches])
            proj_pts = np.float32([proj_pts[m.queryIdx] for m in proj_matches])

            # calculate pixel distance for observed points that are close to projected pts
            dist = np.linalg.norm(proj_pts - obs, axis=1)
            dist_mask = dist < 100

            # list of which map points are tracked
            tracked_map_pts = [proj_map_pts[m.queryIdx] for i, m in enumerate(proj_matches) if dist_mask[i]]

            # get a set of feature indicies which are tracked in the current frame
            tracked_pts_idx = [m.trainIdx for i, m in enumerate(proj_matches) if dist_mask[i]]
            tracked_pts_idx_set = set(tracked_pts_idx)

            # update mask to leave out points that are already tracked
            for i, m in enumerate(matches):
                if m.trainIdx in tracked_pts_idx_set:
                    mask[i] = [0]
            
            # triangulate matched points that does not have corresponding 3d map point using updated mask
            pts3d, idx1, idx2 = triangulate_points(kp1, kp2, matches, rel_pose, mask, self.K)

            # transform triangulated pts to current frame
            pts3d = pts3d.squeeze()
            pts3d_h = np.hstack([pts3d, np.ones((pts3d.shape[0], 1))])
            pts3d = (curr_pose @ pts3d_h.T).T[:, :3]

            # add current frame to map
            self.slam_map.add_frame(curr_pose, kp2, des2)
            curr_frame = self.slam_map.frames[-1]
            
            # add triangulated points as map points
            for i, pt in enumerate(pts3d):
                map_pt = MapPoint(pt)
                self.slam_map.add_map_point(map_pt)
                prev_frame.add_observation(map_pt, idx1[i])
                curr_frame.add_observation(map_pt, idx2[i])

            # add tracked map points as observations in current frame
            for i, map_pt in enumerate(tracked_map_pts):
                curr_frame.add_observation(map_pt, tracked_pts_idx[i])
        
        return self.slam_map



