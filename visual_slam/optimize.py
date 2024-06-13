import g2o
import numpy as np
from .structs import SLAMMap
import math


def window_BA(slam_map: SLAMMap, n, K, fix_pts=False, fix_frames=False):
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(solver)


    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    #cam_params = g2o.CameraParameters(fx, (cx, cy), 0)
    cam_params = g2o.CameraParameters(1.0, (0.0, 0.0), 0)
    cam_params.set_id(0)
    optimizer.add_parameter(cam_params)
    
    # extract vertices and edges
    mpids = set()
    fids = set()
    obs = {}
    for i, fid in enumerate(reversed(slam_map.frames)):
        if i == n: break
        fids.add(fid)
        frame = slam_map.frames[fid]
        for mpid, idx in frame.obs.items():
            pt_norm = np.float32(frame.kp[idx].pt)
            pt_norm[0] = (pt_norm[0] - cx) / fx
            pt_norm[1] = (pt_norm[1] - cy) / fy
            obs[(fid, mpid)] = pt_norm
            if not mpid in mpids:
                mpids.add(mpid)

    # add map point vertices
    for mpid in mpids:
        map_pt = slam_map.map_pts[mpid]
        v_pt = g2o.VertexPointXYZ()
        v_pt.set_id(2 * mpid + 1)
        v_pt.set_estimate(map_pt.pt3d)
        v_pt.set_marginalized(True)
        v_pt.set_fixed(fix_pts)
        optimizer.add_vertex(v_pt)

    # add camera frame pose vertices
    for fid in fids:
        v_se3 = g2o.VertexSE3Expmap()
        frame = slam_map.frames[fid]
        se3 = g2o.SE3Quat(frame.pose[:3, :3], frame.pose[:3, 3])
        v_se3.set_estimate(se3)
        v_se3.set_fixed(fid <= 5)
        v_se3.set_id(2 * fid + 2)
        optimizer.add_vertex(v_se3)

    # add edges 
    err_squared = 0
    for (fid, mpid), m in obs.items():
        edge = g2o.EdgeProjectXYZ2UV()
        edge.set_parameter_id(0, 0)
        v_pt = optimizer.vertex(2 * mpid + 1)
        v_se3 = optimizer.vertex(2 * fid + 2)
        edge.set_vertex(0, v_pt)
        edge.set_vertex(1, v_se3)
        edge.set_measurement(m)
        edge.set_information(np.eye(2))
        edge.set_robust_kernel(g2o.RobustKernelHuber(np.sqrt(5.991)))
        optimizer.add_edge(edge)

        err = cam_params.cam_map(v_se3.estimate().map(v_pt.estimate())) - m
        err_squared += np.sum(err * err)

    optimizer.set_verbose(True)
    optimizer.initialize_optimization()
    optimizer.compute_active_errors()
    #assert math.isclose(optimizer.chi2(), err_squared)
    optimizer.optimize(20)

    for mpid in mpids:
        map_pt = slam_map.map_pts[mpid]
        pt_est = optimizer.vertex(2 * mpid + 1).estimate()
        map_pt.pt3d = np.array(pt_est)

    for fid in fids:
        frame = slam_map.frames[fid]
        pose_est = optimizer.vertex(2 * fid + 2).estimate().matrix()
        frame.pose = np.array(pose_est)
