from GUI import GUI
from HAL import HAL

import cv2
import numpy as np
import time


class Reconstruction:
    def __init__(self, image_left: np.ndarray, image_right: np.ndarray, verbose: bool = False):
        self.image_left = image_left
        self.image_right = image_right

        self.h, self.w = self.image_left.shape[:2]

        # Get camera position
        self.point_3d_camleft = np.append(HAL.getCameraPosition('left'), 1)
        self.point_3d_camright = np.append(HAL.getCameraPosition('right'), 1)

        self.verbose = verbose

    def algorithm(self):

        # Get points of interest
        points_left, points_right = self._get_points_interest()
        print(f"Number of points of interest {points_left.shape[0]}")

        for y, x in points_left:

            # Get 3D point
            point_3d_left = self._get_3d_point(camera='left', x=x, y=y)
            point_3d_cam = self.point_3d_camleft

            # Get direction of the 3D line
            dir_3d_line = (point_3d_left - point_3d_cam)

            # Get epilolar mask
            mask = self._get_epipolar_mask(camera='right', direction_3d=dir_3d_line,
                                           point3d_camera=point_3d_cam, thickness=10)

            im1 = self.image_left.copy()
            cv2.circle(im1, (x, y), 2, (0, 255, 0), -1)

            GUI.showImages(im1, mask, True)

        return self.image_left, cv2.Canny(cv2.cvtColor(self.image_left, cv2.COLOR_BGR2GRAY), 100, 200)

    def _get_points_interest(self) -> (np.ndarray, np.ndarray):
        """
        Get points of interest in the images applying the Canny algorithm
        :return: numpy array of points of interest in the left and right image
        """
        edges_left = cv2.Canny(cv2.cvtColor(self.image_left, cv2.COLOR_BGR2GRAY), 100, 200)
        edges_right = cv2.Canny(cv2.cvtColor(self.image_right, cv2.COLOR_BGR2GRAY), 100, 200)

        return np.argwhere(edges_left == 255), np.argwhere(edges_right == 255)

    def _get_3d_point(self, camera: str, x: float, y: float) -> np.ndarray:
        """
        Get 3D point from 2D point
        :param camera: camera (left or right)
        :param x: x coordinate in the image
        :param y: y coordinate in the image
        :return: 3D point
        """
        # Transform the Image Coordinate System to the Camera System
        point_2d_left = HAL.graficToOptical(camera, [x, y, 1])
        # Backproject a 3D Point Space into the 2D Image Point
        point_3d_left = HAL.backproject(camera, point_2d_left)

        return point_3d_left

    def _get_2d_point(self, camera: str, point_3d: np.ndarray) -> np.ndarray:
        """
        Get 2D point from 3D point
        :param camera: camera (left or right)
        :param point_3d: 3D point
        :return: 2D point
        """
        # Back project a 3D Point Space into the 2D Image Point
        point_2d = HAL.project(camera, point_3d)
        # Transform the Camera System to the Image Coordinate System
        point_2d = HAL.opticalToGrafic(camera, point_2d)

        return point_2d

    def _get_epipolar_mask(self, camera: str, direction_3d: np.ndarray, point3d_camera: np.ndarray, thickness: int = 8) -> np.ndarray:
        """
        Get epipolar mask
        :param camera: camera (left or right)
        :param direction_3d: direction of the 3D line
        :param point3d_camera: 3D point from the camera
        :param thickness: thickness of the epipolar line
        :return: mask with the epipolar line
        """

        # Get 2D points projected(pixels) in the image (left or right) from the 3D line
        point1_2d_right = self._get_2d_point(camera, direction_3d * 5)
        point2_2d_right = self._get_2d_point(camera, direction_3d + point3d_camera)

        point1_line, point2_line = self._get_point_line(point1_2d_right, point2_2d_right)

        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.line(mask, point1_line, point2_line, 255, thickness)

        return mask

    def _get_point_line(self, p1: np.ndarray, p2: np.ndarray) -> (int, int):
        m = (p2[1] - p1[1]) / (p2[0] - p1[0]) + 1e-6
        b = p1[1] - m * p1[0]

        y_final = m * self.w + b
        x_final = (self.h - b) / m

        return (0, int(m * 0 + b)), (int(x_final), self.h) if x_final <= self.w else (self.w, int(y_final))


while True:
    reconstruction = Reconstruction(HAL.getImage('left'), HAL.getImage('right'), verbose=True)

    out_left, out_right = reconstruction.algorithm()

    GUI.showImages(out_left, out_right, True)

    print("hola")
    time.sleep(10)
