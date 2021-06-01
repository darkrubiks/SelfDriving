import os

import cv2
import numpy as np
import glob
import pickle


class CameraCalibration:
    def __init__(self, chessboard_size, square_size, images_path):
        self.chessboard_size = chessboard_size  # Defining the dimensions of checkerboard
        self.square_size = square_size
        self.images_path = images_path
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Creating vector to store vectors of 3D points for each checkerboard image
        self.objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        self.imgpoints = []
        # Defining the world coordinates for 3D points
        self.objp = np.zeros((1, self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        self.objp = self.objp * self.square_size

    def get_matrix(self, drawChess=False):
        # Extracting path of individual image stored in a given directory
        images = glob.glob(self.images_path + '/*.jpg')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                     cv2.CALIB_CB_FAST_CHECK +
                                                     cv2.CALIB_CB_NORMALIZE_IMAGE)
            # If desired number of corner are detected,we refine the pixel coordinates
            if ret:
                self.objpoints.append(self.objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.imgpoints.append(corners2)
                print(f'Found ChessBoard Pattern on image: {fname}')

                if drawChess:
                    # If drawChess draw the pattern found and save image
                    drawImg = cv2.drawChessboardCorners(img, self.chessboard_size, corners2, ret)
                    if not os.path.isdir('chess_pattern'):
                        os.mkdir('chess_pattern')
                    cv2.imwrite('.\\chess_pattern\\' + os.path.basename(fname), drawImg)
            else:
                print(f'Could not find ChessBoard Pattern on image: {fname}')

        if self.imgpoints:
            # Get camera matrix
            _, mtx, dist, _, _ = cv2.calibrateCamera(self.objpoints, self.imgpoints,
                                                  gray.shape[::-1], None, None)

            return mtx, dist

        else:
            return 0


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
    path = '.\\images'
    calib = CameraCalibration((5, 7), 1, path)
    mtx, dist = calib.get_matrix(True)

    with open('mtx.p', 'wb') as fp:
        pickle.dump(mtx.tolist(), fp)

    with open('dist.p', 'wb') as fp:
        pickle.dump(dist.tolist(), fp)

    images = glob.glob(path + '/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        if not os.path.isdir('undistort'):
            os.mkdir('undistort')
        cv2.imwrite('.\\undistort\\' + os.path.basename(fname), undistort(img, mtx, dist))