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
        # self.point_3d_camright = np.append(HAL.getCameraPosition('right'), 1)

        self.verbose = verbose

    def algorithm(self):

        # Get points of interest
        points_left, points_right = self._get_points_interest()
        print(f"Number of points of interest {points_left.shape[0]}")

        for y, x in points_left:

            im1 = self.image_left.copy()
            im2 = self.image_right.copy()

            # if self.verbose:
            #     cv2.circle(self.image_left, (x, y), 10, (255, 0, 0), 10)

            # Get 3D point
            point_3d_left = self._get_3d_point('left', x, y)

            dir = (point_3d_left - self.point_3d_camleft)

            point1_2d_right = self._get_2d_point('right', dir * 5)
            point2_2d_right = self._get_2d_point('right', dir + self.point_3d_camleft)

            if self.verbose:
                print(f"Point 1 image right : {point1_2d_right}; Point 2 image right: {point2_2d_right}")

            def line(p1, p2):
                m = (p2[1] - p1[1]) / (p2[0] - p1[0]) + 0.0000001
                b = p1[1] - m * p1[0]

                y_inicio = m * 0 + b
                y_final = m * self.w + b
                x_inicio = (0 - b) / m
                x_final = (self.h - b) / m
                print(y_inicio, y_final, x_final, x_inicio)

                # Encontrar los puntos de inicio y finales
                punto_inicio = (0, int(y_inicio))
                punto_final = (int(x_final), self.h) if x_final <= self.w else (self.w, int(y_final))

                return punto_inicio, punto_final

            punto_inicio, punto_final = line(point1_2d_right, point2_2d_right)

            cv2.circle(im1, (x, y), 2, (0, 255, 0), -1)
            cv2.line(im2, punto_inicio, punto_final, (0, 255, 0), 1)

            GUI.showImages(im1, im2, True)
            time.sleep(5)

            # if self.verbose:
            #     print(f"Point 1 image right : {point1_2d_right}; Point 2 image right: {point2_2d_right}")

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
        # Backprojects a 3D Point Space into the 2D Image Point
        point_2d = HAL.project(camera, point_3d)
        # Transform the Camera System to the Image Coordinate System
        point_2d = HAL.opticalToGrafic(camera, point_2d)

        return point_2d


while True:
    reconstruction = Reconstruction(HAL.getImage('left'), HAL.getImage('right'), verbose=True)

    out_left, out_right = reconstruction.algorithm()

    GUI.showImages(out_left, out_right, True)

    print("hola")
    time.sleep(10)
