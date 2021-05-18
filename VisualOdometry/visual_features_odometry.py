import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_features(image):
    """
    Find keypoints and descriptors for the image

    Arguments:
    image -- a grayscale image

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=5000)
    kp, des = orb.detectAndCompute(image, None)
    return kp, des


def visualize_features(image, kp):
    """
    Visualize extracted features in the image

    Arguments:
    image -- a grayscale image
    kp -- list of the extracted keypoints

    Returns:
    """
    display = cv2.drawKeypoints(image, kp, None)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(display)


def match_features(des1, des2):
    """
    Match features from two images

    Arguments:
    des1 -- list of the keypoint descriptors in the first image
    des2 -- list of the keypoint descriptors in the second image

    Returns:
    match -- list of matched features from two images. Each match[i] is k or less matches for the same query descriptor
    """
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    match = bf.knnMatch(des1, des2, k=2)
    return match


def filter_matches_distance(match, dist_threshold):
    """
    Filter matched features from two images by distance between the best matches

    Arguments:
    match -- list of matched features from two images
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0)

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold
    """
    filtered_match = []
    for m, n in match:
        if m.distance < dist_threshold * n.distance:
            filtered_match.append(m)
    return filtered_match


def visualize_matches(image1, kp1, image2, kp2, match):
    """
    Visualize corresponding matches in two images

    Arguments:
    image1 -- the first image in a matched image pair
    kp1 -- list of the keypoints in the first image
    image2 -- the second image in a matched image pair
    kp2 -- list of the keypoints in the second image
    match -- list of matched features from the pair of images

    Returns:
    image_matches -- an image showing the corresponding matches on both image1 and image2 or None if you don't use this function
    """
    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, match, None, flags=2)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)


def estimate_motion(match, kp1, kp2, k):
    """
    Estimate camera motion from a pair of subsequent image frames

    Arguments:
    match -- list of matched features from the pair of images
    kp1 -- list of the keypoints in the first image
    kp2 -- list of the keypoints in the second image
    k -- camera calibration matrix

    Returns:
    rmat -- recovered 3x3 rotation numpy matrix
    tvec -- recovered 3x1 translation numpy vector
    image1_points -- a list of selected match coordinates in the first image. image1_points[i] = [u, v], where u and v are
                     coordinates of the i-th match in the image coordinate system
    image2_points -- a list of selected match coordinates in the second image. image1_points[i] = [u, v], where u and v are
                     coordinates of the i-th match in the image coordinate system

    """
    image1_points = []
    image2_points = []

    for m in match:
        train_idx = m.trainIdx
        query_idx = m.queryIdx

        p1x, p1y = kp1[query_idx].pt
        image1_points.append([p1x, p1y])

        p2x, p2y = kp2[train_idx].pt
        image2_points.append([p2x, p2y])

    E, mask = cv2.findEssentialMat(np.array(image1_points), np.array(image2_points), k)

    retval, rmat, tvec, mask = cv2.recoverPose(E, np.array(image1_points),
                                               np.array(image2_points), k,
                                               cv2.RANSAC, 0.999, 1.0)

    return rmat, tvec, image1_points, image2_points


def estimate_trajectory(match, kp1, kp2, k):
    """
    Estimate complete camera trajectory from subsequent image pairs

    Arguments:
    estimate_motion -- a function which estimates camera motion from a pair of subsequent image frames
    match -- matches for each subsequent image pair.
    kp1,kp2 --  keypoints from image
    k -- camera calibration matrix

    Returns:
    trajectory -- a numpy array of the camera locations, trajectory[:, i] is a 3x1 numpy vector, such as:

                  trajectory[:, i][0] - is X coordinate of the i-th location
                  trajectory[:, i][1] - is Y coordinate of the i-th location
                  trajectory[:, i][2] - is Z coordinate of the i-th location

    """

    R = np.diag([1, 1, 1])
    T = np.zeros([3, 1])
    RT = np.hstack([R, T])
    RT = np.vstack([RT, np.zeros([1, 4])])
    RT[-1, -1] = 1

    rmat, tvec, image1_points, image2_points = estimate_motion(match, kp1, kp2, k)
    rt_mtx = np.hstack([rmat, tvec])
    rt_mtx = np.vstack([rt_mtx, np.zeros([1, 4])])
    rt_mtx[-1, -1] = 1

    rt_mtx_inv = np.linalg.inv(rt_mtx)

    RT = np.dot(RT, rt_mtx_inv)
    trajectory = RT[:3, 3]

    return trajectory


if __name__ == '__main__':

    with open('CameraCalibration/camera_data.p', 'rb') as fp:
        k = pickle.load(fp)  # load camera matrix

    trajectory = [np.array([0, 0, 0])]

    filter_match = True

    path = ''  # video path

    cap = cv2.VideoCapture(path)
    ret, previous_frame = cap.read()
    pre_kp, pre_des = extract_features(previous_frame)  # extract features and descriptor from previous image
    while cap.isOpened():
        ret, next_frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            break
        # extract features and descriptor from next image
        next_kp, next_des = extract_features(next_frame)
        # match descriptors
        match = match_features(pre_des, next_des)
        # filter match
        if filter_match:
            match = filter_matches_distance(match, 0.6)
        # estimate trajectory
        new_trajectory = estimate_trajectory(match, pre_kp, next_kp, k)
        # append to trajectory list
        trajectory.append(new_trajectory)
        # re-use the last frame now as the previous frame
        pre_kp, pre_des = next_kp, next_des

    cap.release()
    trajectory = np.array(trajectory).T
