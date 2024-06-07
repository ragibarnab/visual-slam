import cv2
import numpy as np
import pangolin  # Assuming you have a Pangolin wrapper for Python
import OpenGL.GL as gl
import glob

# Initialize the ORB detector
orb = cv2.ORB_create()

# Create a BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# '/home/rae384/data/kitti/odometry/sequences/00/calib.txt'
calib_file = '/home/rae384/data/kitti/odometry/dataset/sequences/00/calib.txt'
with open(calib_file, 'r') as f:
    proj_mat = f.readline()
    proj_mat = proj_mat.strip().split(' ')[1:]
    proj_mat = np.float32(proj_mat).reshape(3, 4)
K = cv2.decomposeProjectionMatrix(proj_mat)[0]
fx = K[0,0]
fy = K[1,1]
cx = K[0,2]
cy = K[1,2]


def detect_and_compute(image):
    """ Detects and computes ORB keypoints and descriptors. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_features(des1, des2, threshold=50):
    """ Matches features between two sets of descriptors with a distance threshold. """
    matches = bf.match(des1, des2)
    matches = [m for m in matches if m.distance < threshold]
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def pose_estimation(kp1, kp2, matches):
    """ Estimates the pose using matched keypoints and returns inliers mask. """
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    E, mask = cv2.findEssentialMat(pts1, pts2, K)
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    return R, t, mask_pose

def triangulate_points(kp1, kp2, matches, R, t, mask, global_pose):
    """ Triangulates 3D points from matched keypoints using inliers mask and transforms them to global frame. """
    pts1 = np.float32([kp1[m.queryIdx].pt for i, m in enumerate(matches) if mask[i]])
    pts2 = np.float32([kp2[m.trainIdx].pt for i, m in enumerate(matches) if mask[i]])
    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3).T
    pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1, 3).T

    P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(K, np.hstack((R, t)))

    points_4d_h = cv2.triangulatePoints(P1, P2, pts1_h[:2], pts2_h[:2])
    points_3d = cv2.convertPointsFromHomogeneous(points_4d_h.T).reshape(-1, 3)

    # Transform the points to the global frame using the accumulated pose
    points_3d_h = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    points_3d_global = (global_pose @ points_3d_h.T).T[:, :3]
    
    return points_3d_global

def extract_descriptors_from_keypoints(keypoints, descriptors, points):
    """ Extracts descriptors for the given keypoints corresponding to 2D points. """
    extracted_descriptors = []
    for point in points:
        for i, kp in enumerate(keypoints):
            if np.allclose(kp.pt, point, atol=5):  # 5-pixel tolerance
                extracted_descriptors.append(descriptors[i])
                break
    return np.array(extracted_descriptors)

# /home/rae384/data/kitti/odometry/sequences/00/image_0/
img_seq_dir = '/home/rae384/data/kitti/odometry/dataset/sequences/00/image_0/'
img_seq = sorted(glob.glob(img_seq_dir + '*.png'))
print(img_seq[0])

# Read the first frame
frame1 = cv2.imread(img_seq[0])
kp1, des1 = detect_and_compute(frame1)

# Initialize the map with keyframes and landmarks
keyframes = [frame1]
landmarks = []

# Initialize the global pose (identity matrix)
global_pose = np.eye(4)
poses = [global_pose.copy()]

# Initialize Pangolin visualization
pangolin.CreateWindowAndBind('Map', 640, 480)
gl.glEnable(gl.GL_DEPTH_TEST)

# Define camera projection and modelview matrices
s_cam = pangolin.OpenGlRenderState(
    pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
    pangolin.ModelViewLookAt(0, -10, -10, 0, 0, 0, pangolin.AxisY)
)

# Create Interactive View in Pangolin
d_cam = pangolin.CreateDisplay()
d_cam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
d_cam.SetHandler(pangolin.Handler3D(s_cam))

# Main loop
for img_file in img_seq[1:]:
    frame2 = cv2.imread(img_file)

    # Detect and compute ORB features for the current frame
    kp2, des2 = detect_and_compute(frame2)

    # Match features between the previous and the current frame
    matches = match_features(des1, des2)

    # Estimate pose and get the inliers mask
    R, t, mask = pose_estimation(kp1, kp2, matches)

    # Accumulate the global pose
    current_pose = np.eye(4)
    current_pose[:3, :3] = R
    current_pose[:3, 3] = t.squeeze()
    global_pose = global_pose @ current_pose
    poses.append(global_pose.copy())

    # Triangulate new points using inliers mask and add them to the landmarks
    new_landmarks = triangulate_points(kp1, kp2, matches, R, t, mask, global_pose)
    landmarks.append(new_landmarks)

    # Update the previous frame keypoints and descriptors
    kp1, des1 = kp2, des2
    frame1 = frame2.copy()

    # Visualize the camera trajectory and landmarks in Pangolin
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    #gl.glClearColor(1.0, 1.0, 1.0, 1.0)
    d_cam.Activate(s_cam)

    # Draw the camera trajectory
    #gl.glColor3f(1.0, 0.0, 0.0)
    #gl.glBegin(gl.GL_LINE_STRIP)
    for i in range(len(poses)):
        #pose = global_pose[:3, 3]
        #gl.glVertex3f(pose[0], pose[1], pose[2])
        gl.glLineWidth(0.1)
        gl.glColor3f(0.0, 0.0, 1.0)
        pangolin.DrawCamera(poses[i], 0.25, 0.5, 0.5)
    #gl.glEnd()

    # Draw the landmarks
    gl.glColor3f(0.0, 1.0, 0.0)
    gl.glBegin(gl.GL_POINTS)
    for landmark_set in landmarks:
        for landmark in landmark_set:
            gl.glVertex3f(landmark[0], landmark[1], landmark[2])
    gl.glEnd()

    pangolin.FinishFrame()

    cv2.waitKey(1000)
    cv2.imshow('window', frame2)

cv2.destroyAllWindows()

