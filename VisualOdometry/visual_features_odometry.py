import math
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec


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
    orb = cv2.ORB_create(nfeatures=1500)
    kp, des = orb.detectAndCompute(image, None)
    return kp, des


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
                                               np.array(image2_points), k)

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


def visualize_trajectory(trajectory):
    # Unpack X Y Z each trajectory point
    locX = []
    locY = []
    locZ = []
    # This values are required for keeping equal scale on each plot.
    # matplotlib equal axis may be somewhat confusing in some situations because of its various scale on
    # different axis on multiple plots
    max = -math.inf
    min = math.inf

    # Needed for better visualisation
    maxY = -math.inf
    minY = math.inf

    for i in range(0, trajectory.shape[1]):
        current_pos = trajectory[:, i]

        locX.append(current_pos.item(0))
        locY.append(current_pos.item(1))
        locZ.append(current_pos.item(2))
        if np.amax(current_pos) > max:
            max = np.amax(current_pos)
        if np.amin(current_pos) < min:
            min = np.amin(current_pos)

        if current_pos.item(1) > maxY:
            maxY = current_pos.item(1)
        if current_pos.item(1) < minY:
            minY = current_pos.item(1)

    auxY_line = locY[0] + locY[-1]
    if max > 0 and min > 0:
        minY = auxY_line - (max - min) / 2
        maxY = auxY_line + (max - min) / 2
    elif max < 0 and min < 0:
        minY = auxY_line + (min - max) / 2
        maxY = auxY_line - (min - max) / 2
    else:
        minY = auxY_line - (max - min) / 2
        maxY = auxY_line + (max - min) / 2

    # Set styles
    mpl.rc("figure", facecolor="white")
    plt.style.use("seaborn-whitegrid")

    # Plot the figure
    fig = plt.figure(figsize=(8, 6), dpi=100)
    gspec = gridspec.GridSpec(3, 3)
    ZY_plt = plt.subplot(gspec[0, 1:])
    YX_plt = plt.subplot(gspec[1:, 0])
    traj_main_plt = plt.subplot(gspec[1:, 1:])
    D3_plt = plt.subplot(gspec[0, 0], projection='3d')

    # Actual trajectory plotting ZX
    toffset = 1.06
    traj_main_plt.set_title("Trajectory (Z, X)", y=toffset)
    traj_main_plt.set_title("Trajectory (Z, X)", y=1)
    traj_main_plt.plot(locZ, locX, ".-", label="Trajectory", zorder=1, linewidth=1, markersize=4)
    traj_main_plt.set_xlabel("Z")
    # traj_main_plt.axes.yaxis.set_ticklabels([])
    # Plot reference lines
    traj_main_plt.plot([locZ[0], locZ[-1]], [locX[0], locX[-1]], "--", label="Auxiliary line", zorder=0, linewidth=1)
    # Plot camera initial location
    traj_main_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    traj_main_plt.set_xlim([min, max])
    traj_main_plt.set_ylim([min, max])
    traj_main_plt.legend(loc=1, title="Legend", borderaxespad=0., fontsize="medium", frameon=True)

    # Plot ZY
    # ZY_plt.set_title("Z Y", y=toffset)
    ZY_plt.set_ylabel("Y", labelpad=-4)
    ZY_plt.axes.xaxis.set_ticklabels([])
    ZY_plt.plot(locZ, locY, ".-", linewidth=1, markersize=4, zorder=0)
    ZY_plt.plot([locZ[0], locZ[-1]], [(locY[0] + locY[-1]) / 2, (locY[0] + locY[-1]) / 2], "--", linewidth=1, zorder=1)
    ZY_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    ZY_plt.set_xlim([min, max])
    ZY_plt.set_ylim([minY, maxY])

    # Plot YX
    # YX_plt.set_title("Y X", y=toffset)
    YX_plt.set_ylabel("X")
    YX_plt.set_xlabel("Y")
    YX_plt.plot(locY, locX, ".-", linewidth=1, markersize=4, zorder=0)
    YX_plt.plot([(locY[0] + locY[-1]) / 2, (locY[0] + locY[-1]) / 2], [locX[0], locX[-1]], "--", linewidth=1, zorder=1)
    YX_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    YX_plt.set_xlim([minY, maxY])
    YX_plt.set_ylim([min, max])

    # Plot 3D
    D3_plt.set_title("3D trajectory", y=toffset)
    D3_plt.plot3D(locX, locZ, locY, zorder=0)
    D3_plt.scatter(0, 0, 0, s=8, c="red", zorder=1)
    D3_plt.set_xlim3d(min, max)
    D3_plt.set_ylim3d(min, max)
    D3_plt.set_zlim3d(min, max)
    D3_plt.tick_params(direction='out', pad=-2)
    D3_plt.set_xlabel("X", labelpad=0)
    D3_plt.set_ylabel("Z", labelpad=0)
    D3_plt.set_zlabel("Y", labelpad=-2)

    # plt.axis('equal')
    D3_plt.view_init(45, azim=30)
    plt.tight_layout()
    plt.show()


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
    image_matches = cv2.resize(image_matches, (1280, 720))
    return image_matches


def visualize_features(image, kp):
    """
    Visualize extracted features in the image

    Arguments:
    image -- a grayscale image
    kp -- list of the extracted keypoints

    Returns:
    """
    display = cv2.drawKeypoints(image, kp, None)
    display = cv2.resize(display, (1280, 720))
    return display


def visualize_camera_movement(image1, image1_points, image2, image2_points, is_show_img_after_move=False):
    image1 = image1.copy()
    image2 = image2.copy()

    for i in range(0, len(image1_points)):
        # Coordinates of a point on t frame
        p1 = (int(image1_points[i][0]), int(image1_points[i][1]))
        # Coordinates of the same point on t+1 frame
        p2 = (int(image2_points[i][0]), int(image2_points[i][1]))

        cv2.circle(image1, p1, 5, (0, 255, 0), 1)
        cv2.arrowedLine(image1, p1, p2, (0, 255, 0), 1)
        cv2.circle(image1, p2, 5, (255, 0, 0), 1)

        if is_show_img_after_move:
            cv2.circle(image2, p2, 5, (255, 0, 0), 1)

    if is_show_img_after_move:
        image2 = cv2.resize(image2, (1280, 720))
        return image2
    else:
        image1 = cv2.resize(image1, (1280, 720))
        return image1


def undistort(image, mtx, dist):
    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # undistort
    dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst


if __name__ == '__main__':

    with open('C:\\Users\\leona\\Desktop\\SelfDriving\\CameraCalibration\\mtx.p', 'rb') as fp:
        mtx = pickle.load(fp)  # load camera matrix
        mtx = np.array(mtx)

    with open('C:\\Users\\leona\\Desktop\\SelfDriving\\CameraCalibration\\dist.p', 'rb') as fp:
        dist = pickle.load(fp)  # load camera matrix
        dist = np.array(dist)

    trajectory = [np.array([0, 0, 0])]

    filter_match = True

    path = 'C:\\Users\\leona\\Desktop\\SelfDriving\\VisualOdometry\\GOPR1268.MP4'  # video path

    cap = cv2.VideoCapture(path)
    ret, previous_frame = cap.read()
    previous_frame = undistort(previous_frame, mtx, dist)
    pre_kp, pre_des = extract_features(previous_frame)  # extract features and descriptor from previous image
    while cap.isOpened():
        ret, next_frame = cap.read()
        next_frame = undistort(next_frame, mtx, dist)
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

        rmat, tvec, image1_points, image2_points = estimate_motion(match, pre_kp, next_kp, mtx)

        cv2.imshow('bla', visualize_camera_movement(previous_frame, image1_points, next_frame, image2_points))
        if cv2.waitKey(1) == ord('q'):
            break

        # estimate trajectory
        new_trajectory = estimate_trajectory(match, pre_kp, next_kp, mtx)
        # append to trajectory list
        trajectory.append(new_trajectory)
        # re-use the last frame now as the previous frame
        pre_kp, pre_des = next_kp, next_des
        previous_frame = next_frame

    cap.release()
    trajectory = np.array(trajectory).T
    visualize_trajectory(trajectory)
