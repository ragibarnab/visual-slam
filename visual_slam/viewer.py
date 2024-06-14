import OpenGL.GL as gl
import pangolin
import numpy as np
from .structs import SLAMMap
from multiprocessing import Process, Queue

# cam_to_world = np.float64([
#     [0, -1, 0, 0],
#     [0, 0, -1, 0],
#     [-1, 0, 0, 0],
#     [0, 0, 0, 1]
# ])


class Viewer():
    ''' credits to george hotz'''

    def __init__(self):
        self.state = None
        self.q = Queue()
        self.p = Process(target=self.viewer_process, args=(self.q,))
        self.p.daemon = True
        self.p.start()


    def init_viewer(self):
        pangolin.CreateWindowAndBind('Main', 1241, 376)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
            pangolin.ModelViewLookAt(0, 0, -2, 0, 0, 0, pangolin.AxisDirection.AxisNegY))
        handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.dcam.SetHandler(handler)
        # hack to avoid small Pangolin, no idea why it's *2
        self.dcam.Resize(pangolin.Viewport(0,0,640*2,480*2))
        self.dcam.Activate()


    def viewer_process(self, q):
        self.init_viewer()
        while True:
            self.refresh_viewer(q)


    def refresh_viewer(self, q: Queue):
        while not q.empty():
            self.state = q.get()
        
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.dcam.Activate(self.scam)

        if self.state is not None:

            points, frames = self.state

            # Draw Point Cloud 
            if points:
                points = np.array(points)
                #points = (cam_to_world[:3, :3] @ points.T).T
                gl.glPointSize(1.0)
                gl.glColor3f(1.0, 0.0, 0.0)
                pangolin.DrawPoints(points)

            for f in frames:
                gl.glLineWidth(0.1)
                gl.glColor3f(0.0, 0.0, 1.0)
                pose = np.linalg.inv(f)
                #pose = f
                pangolin.DrawCamera(pose, 0.25, 0.5, 0.5)
                    
        pangolin.FinishFrame()
    

    def update(self, slam_map: SLAMMap):

        if not self.q:
            return

        points = []
        for mp in slam_map.map_pts.values():
            points.append(mp.pt3d)

        frames = []
        for f in slam_map.frames.values():
            frames.append(f.pose)

        self.q.put((points, frames))
        