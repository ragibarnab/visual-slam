# Monocular Visual SLAM

![Demo](/monocular_visual_slam.gif)


# Software used
* [opencv-python](https://pypi.org/project/opencv-python/): computer vision algorithms and 2D visualization
* [pangolin](https://github.com/uoip/pangolin): 3D visualization. See [this](https://github.com/stevenlovegrove/Pangolin/pull/318/files) for correctly compiling.
* [g2o-python](https://pypi.org/project/g2o-python/): bundle adjustment and pose graph optimization



## TO-DOs
* Track map points across frames by projection for better optimization
* Remove points not trackable across frames
* Improve local window BA
* Implement place recognition for loop closing