import numpy as np
import cv2
from .structs import SLAMMap, Frame, MapPoint
from .utils import Rt_to_pose, pose_to_Rt, project_points
from .feature_extraction import match_features, extract_features
from .pose_estimation import estimate_pose, estimate_pose_cv2
from .triangulation import triangulate_points_lls, triangulate_points
from .optimize import window_BA


class VisualSLAM():

    def __init__(self, K) -> None:
        self.K = K
        self.slam_map = None
        self.initialized = False

    def process_frame(self, frame) -> SLAMMap:

        if not self.initialized:
            if self.slam_map is None:
                kp, des = extract_features(frame)
                self.slam_map = SLAMMap()
                self.slam_map.add_frame(np.eye(4), kp, des)
            else:
                prev_frame = self.slam_map.get_last_frame()
                
                kp1, des1 = prev_frame.kp, prev_frame.des
                kp2, des2 = extract_features(frame)
                
                matches = match_features(des1, des2)

                R, t, mask = estimate_pose_cv2(kp1, kp2, matches, self.K)
                rel_pose = Rt_to_pose(R, t)

                pts3d, idx1, idx2 = triangulate_points(kp1, kp2, matches, prev_frame.pose, rel_pose, mask, self.K)

                # add current frame to map
                curr_frame = self.slam_map.add_frame(rel_pose, kp2, des2)

                # add observations and map points
                for i, pt in enumerate(pts3d):
                    map_pt = self.slam_map.add_map_point(pt)
                    prev_frame.add_observation(map_pt.mpid, idx1[i])
                    curr_frame.add_observation(map_pt.mpid, idx2[i])


                #window_BA(self.slam_map, 2, self.K)

                self.initialized = True
        else:
            # compute kp, des for current frame
            prev_frame = self.slam_map.get_last_frame()
            kp1, des1 = prev_frame.kp, prev_frame.des
            kp2, des2 = extract_features(frame)

            matches = match_features(des1, des2)
            
            # return cam2 to cam1 Rt
            R, t, mask = estimate_pose_cv2(kp1, kp2, matches, self.K)
            rel_pose = Rt_to_pose(R, t)
            curr_pose = rel_pose @ prev_frame.pose

            # # get "visible" map points and their descriptors
            # visible = set()
            # proj_map_pts = []
            # proj_des = []
            # for i, fid in enumerate(reversed(self.slam_map.frames)):
            #     if i == 8: break
            #     frame = self.slam_map.frames[fid]
            #     for mpid, idx in frame.obs.items():
            #         if mpid not in visible:
            #             visible.add(mpid)
            #             proj_map_pts.append(self.slam_map.map_pts[mpid])
            #             proj_des.append(frame.des[idx])
            # proj_des = np.uint8(proj_des)

            # # match "visible" map points to current frame's features and project them
            # proj_matches = match_features(proj_des, des2, threshold=50)
            # obs = np.float32([kp2[m.trainIdx].pt for m in proj_matches])
            # to_proj = np.float32([proj_map_pts[m.queryIdx].pt3d for m in proj_matches])
            # proj_pts = project_points(to_proj, curr_pose, self.K)

            # project_points(to_proj)

            # # calculate pixel distance for observed feature points that are close to projected pts
            # dist = np.linalg.norm(proj_pts - obs, axis=1)
            # dist_mask = dist < 20

            # # list of which map points were tracked
            # tracked_map_pts = [proj_map_pts[m.queryIdx] for i, m in enumerate(proj_matches) if dist_mask[i]]
            # print(len(tracked_map_pts))

            # # get a set of feature indicies which are tracked in the current frame
            # tracked_pts_idx = [m.trainIdx for i, m in enumerate(proj_matches) if dist_mask[i]]
            # tracked_pts_idx_set = set(tracked_pts_idx)

            # # update mask to leave out points that are already tracked
            # for i, m in enumerate(matches):
            #     if m.trainIdx in tracked_pts_idx_set:
            #         mask[i] = [0]
            
            #triangulate matched points that does not have corresponding 3d map point using updated mask

            pts3d, idx1, idx2 = triangulate_points(kp1, kp2, matches, prev_frame.pose, curr_pose, mask, self.K)

            # add current frame to map
            curr_frame = self.slam_map.add_frame(curr_pose, kp2, des2)
            
            # add triangulated points as map points
            for i, pt in enumerate(pts3d):
                map_pt = self.slam_map.add_map_point(pt)
                prev_frame.add_observation(map_pt.mpid, idx1[i])
                curr_frame.add_observation(map_pt.mpid, idx2[i])

            # # add tracked map points as observations in current frame
            # for i, map_pt in enumerate(tracked_map_pts):
            #     curr_frame.add_observation(map_pt.mpid, tracked_pts_idx[i])

            # optimize pose
            #window_BA(self.slam_map, 1000, self.K, fix_pts=True)

            # if prev_frame.fid % 8 == 0:
            #     window_BA(self.slam_map, 12, self.K, fix_pts=False)
        
        return self.slam_map



