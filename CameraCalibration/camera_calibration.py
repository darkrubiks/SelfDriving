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
                    drawImg = cv2.drawChessboardCorners(img, self.chessboard_size, corners2, ret)
                    if not os.path.isdir('chess_pattern'):
                        os.mkdir('chess_pattern')
                    cv2.imwrite('.\chess_pattern\\' + os.path.basename(fname), drawImg)
            else:
                print(f'Could not find ChessBoard Pattern on image: {fname}')

        if self.imgpoints:
            _, mtx, _, _, _ = cv2.calibrateCamera(self.objpoints, self.imgpoints,
                                                  gray.shape[::-1], None, None)

            return mtx.tolist()

        else:
            return 0


if __name__ == '__main__':
    calib = CameraCalibration((5, 7), 0.015, '.\images')
    camera_data = calib.get_matrix(True)

    with open('camera_data.p', 'wb') as fp:
        pickle.dump(camera_data, fp)