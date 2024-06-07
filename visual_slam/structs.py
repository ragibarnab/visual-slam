import numpy as np
from collections import OrderedDict

class MapPoint():
    def __init__(self, mpid, pt3d) -> None:
        self.mpid = mpid
        self.pt3d = pt3d
    
        

class Frame():
    def __init__(self, fid, pose, kp, des) -> None:
        self.fid = fid
        self.pose = pose
        self.kp = kp
        self.des = des
        self.obs = {}   # maps map point id to index of keypoints, descriptors

    def add_observation(self, mpid, idx):
        self.obs[mpid] = idx




class SLAMMap():
    def __init__(self) -> None:
        self.frames = OrderedDict()
        self.map_pts = OrderedDict()
        self.fid = 0
        self.mpid = 0

    def get_last_frame(self):
        return self.frames[next(reversed(self.frames))]

    def add_frame(self, pose, kp, des, kf=False):
        frame = Frame(self.fid, pose, kp, des)
        self.frames[self.fid] = frame
        self.fid += 1
        return frame

    def add_map_point(self, pt3d):
        map_pt = MapPoint(self.mpid, pt3d)
        self.map_pts[self.mpid] = map_pt
        self.mpid += 1
        return map_pt

    def get_visible_map_points_from_prev_frames(self, n=7):
        '''Return visible map points and their descriptors seen from previous N frames'''
        frames = self.frames[-n:]

        found = set()
        map_pts_ret = []
        des_ret = []
        for frame in frames:
            for i, map_pt in enumerate(frame.map_pts):
                if map_pt not in found:
                    found.add(map_pt)
                    map_pts_ret.append(map_pt)
                    des_ret.append(frame.des[frame.idx[i]])

        return map_pts_ret, np.uint8(des_ret)


    def serialize(self):
        pass

    def deserialize(self):
        pass
