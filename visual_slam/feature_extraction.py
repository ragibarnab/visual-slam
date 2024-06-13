import cv2

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def extract_features(frame):
    '''Copied from george hotz'''
    # detection
    pts = cv2.goodFeaturesToTrack(frame, 3000, qualityLevel=0.01, minDistance=7)

    # extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
    kps, des = orb.compute(frame, kps)
    return kps, des


def match_features(des1, des2, threshold=50):
    """ Matches features between two sets of descriptors with a distance threshold. """
    matches = bf.match(des1, des2)
    matches = [m for m in matches if m.distance < threshold]
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

