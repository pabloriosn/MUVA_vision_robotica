import cv2
import os
import numpy as np


def main():

    # Define the size of the pattern
    pattern_size = (8, 6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 3D points real world coordinates
    obj_points = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
    obj_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    points3d = []
    points2d = []

    path = "photos/calibration_photos/"
    files = os.listdir(path)

    for file in files:

        img = cv2.imread(path + file)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            points3d.append(obj_points)
            points2d.append(corners2)

    cv2.destroyAllWindows()

    ret, matriz, distortion, r_vecs, t_vecs = cv2.calibrateCamera(points3d, points2d, gray.shape[::-1], None, None)

    if ret:
        np.savez('matrix_calibration', matriz, distortion, r_vecs, t_vecs)


if __name__ == '__main__':
    main()
