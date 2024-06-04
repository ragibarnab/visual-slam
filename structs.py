import numpy as np

class MapPoint():
    def __init__(self, pt3d) -> None:
        self.pt3d = pt3d
        

class Frame():
    def __init__(self, fid, pose, kp, des) -> None:
        self.fid = fid
        self.pose = pose
        self.kp = kp
        self.des = des
        self.idx = []   # stores index of keypoints, descriptors of observed map points
        self.map_pts = []   # stores pointers to observed map points

    def add_observation(self, map_pt, idx):
        self.map_pts.append(map_pt)
        self.idx.append(idx)



class SLAMMap():
    def __init__(self) -> None:
        self.frames = []
        self.key_frames = []
        self.map_pts = set()
        self.fid = 0

    def add_frame(self, pose, kp, des, kf=False):
        frame = Frame(self.fid, pose, kp, des)
        self.frames.append(frame)
        if kf:
            self.key_frames.append(frame)
        self.fid += 1

    def add_map_point(self, map_pt):
        self.map_pts.add(map_pt)

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
