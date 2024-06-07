import g2o
import numpy as np
from .structs import SLAMMap


def window_BA(slam_map: SLAMMap, n, K):
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(solver)

    cam_params = g2o.CameraParameters(K[0,0], (K[0, 2], K[1, 2]), 0)
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
            obs[(fid, mpid)] = np.float32(frame.kp[idx].pt)
            if not mpid in mpids:
                mpids.add(mpid)

    # add map point vertices
    for mpid in mpids:
        map_pt = slam_map.map_pts[mpid]
        v_pt = g2o.VertexPointXYZ()
        v_pt.set_id(2 * mpid + 1)
        v_pt.set_estimate(map_pt.pt3d)
        v_pt.set_marginalized(True)
        v_pt.set_fixed(False)
        optimizer.add_vertex(v_pt)

    # add camera frame pose vertices
    for fid in fids:
        v_se3 = g2o.VertexSE3Expmap()
        frame = slam_map.frames[fid]
        se3 = g2o.SE3Quat(frame.pose[:3, :3], frame.pose[:3, 3])
        v_se3.set_estimate(se3)
        v_se3.set_fixed(fid == 0)
        v_se3.set_id(2 * fid + 2)
        optimizer.add_vertex(v_se3)

    # add edges 
    for (fid, mpid), m in obs.items():
        edge = g2o.EdgeProjectXYZ2UV()
        edge.set_parameter_id(0, 0)
        edge.set_vertex(0, optimizer.vertex(2 * mpid + 1))
        edge.set_vertex(1, optimizer.vertex(2 * fid + 2))
        edge.set_measurement(m)
        edge.set_information(np.eye(2))
        edge.set_robust_kernel(g2o.RobustKernelHuber(np.sqrt(5.991)))
        optimizer.add_edge(edge)

    optimizer.set_verbose(False)
    optimizer.initialize_optimization()
    optimizer.optimize(50)


    for mpid in mpids:
        map_pt = slam_map.map_pts[mpid]
        pt_est = optimizer.vertex(2 * mpid + 1).estimate()
        print(pt_est.shape)
        map_pt.pt3d = pt_est

    for fid in fids:
        frame = slam_map.frames[fid]
        pose_est = optimizer.vertex(2 * fid + 2).estimate().matrix()
        frame.pose = pose_est