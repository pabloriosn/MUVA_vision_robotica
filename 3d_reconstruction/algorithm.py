from GUI import GUI
from HAL import HAL

import cv2
import numpy as np
import time


class Reconstruction:
    def __init__(self, image_left: np.ndarray, image_right: np.ndarray, verbose: bool = False):
        self.image_left = image_left
        self.image_right = image_right

        # Get camera position
        self.camera_left = HAL.getCameraPosition('left')
        self.camera_right = HAL.getCameraPosition('right')

        self.verbose = verbose

    def get_points_interest(self) -> (np.ndarray, np.ndarray):
        edges_left = cv2.Canny(cv2.cvtColor(self.image_left, cv2.COLOR_BGR2GRAY), 100, 200)
        edges_right = cv2.Canny(cv2.cvtColor(self.image_right, cv2.COLOR_BGR2GRAY), 100, 200)

        return np.argwhere(edges_left == 255), np.argwhere(edges_right == 255)

    def algorithm(self):

        # Get points of interest
        points_left, points_right = self.get_points_interest()
        print(f"Number of points of interest {points_left.shape[0]}")

        for y, x in points_left:

            if self.verbose:
                cv2.circle(self.image_left, (x, y), 2, (0, 255, 0), 1)

        return self.image_left, cv2.Canny(cv2.cvtColor(self.image_left, cv2.COLOR_BGR2GRAY), 100, 200)


while True:
    reconstruction = Reconstruction(HAL.getImage('left'), HAL.getImage('right'), verbose=True)

    out_left, out_right = reconstruction.algorithm()

    GUI.showImages(out_left, out_right, True)

    print("hola")
    time.sleep(10)
