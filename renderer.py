import numpy as np
import OpenGL.GL as gl
import pangolin
from structs import SLAMMap


class Renderer():
    def __init__(self) -> None:
        pass

    def render(self, slam_map: SLAMMap):
        if not pangolin.ShouldQuit():
            return
