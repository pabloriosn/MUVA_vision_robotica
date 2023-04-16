from GUI import GUI
from HAL import HAL

import cv2
import numpy as np


class Reconstruction:
    def __init__(self, image_left: np.ndarray, image_right: np.ndarray,
                 camera_origin: str, camera_target: str, verbose: bool = False):

        self.camera_origin = camera_origin
        self.camera_target = camera_target

        self.h, self.w = image_left.shape[:2]
        self.verbose = verbose

        self.images = {
            'right': image_right,
            'left': image_left}

        self.images_hsv = {
            'right': cv2.cvtColor(image_right, cv2.COLOR_BGR2HSV),
            'left': cv2.cvtColor(image_left, cv2.COLOR_BGR2HSV)}

        # Get camera position
        self.pos_3d_cam = {
            'right': np.append(HAL.getCameraPosition('right'), 1),
            'left': np.append(HAL.getCameraPosition('left'), 1)
        }

    def algorithm(self):
        # Get points of interest
        points_interest = self._get_points_interest(self.camera_origin)
        print(f"Number of points of interest {points_interest.shape[0]}")

        for y, x in points_interest:

            # Get 3D points
            point_3d = self._get_3d_point(camera=self.camera_origin, x=x, y=y)

            # Get direction of the 3D line
            dir_3d_line = (point_3d - self.pos_3d_cam[self.camera_origin])

            # Get epilolar mask
            mask = self._get_epipolar_mask(camera=self.camera_target, direction_3d=dir_3d_line,
                                           point3d_camera=self.pos_3d_cam[self.camera_origin], thickness=10)

            # Get homologue point
            point_2d_homologue = self._get_point_homologue(cam_o=self.camera_origin, cam_t=self.camera_target,
                                                           mask=mask, point_2d=(x, y), window_size=10)

            if self.verbose:
                print(f"Punto 1: ({x},{y}) y su homologo {point_2d_homologue}")

            # Get 3d from homologue point
            point_3d_homologue = self._get_3d_point(camera=self.camera_target,
                                                    x=point_2d_homologue[0], y=point_2d_homologue[1])
            # Get direction of the 3D line
            dir_3d_line_homologue = (point_3d_homologue - self.pos_3d_cam[self.camera_target])

            # Triangulation to get 3D point
            point_3d_result = self._get_triangulation(cam_o=self.camera_origin, cam_t=self.camera_target,
                                                      dir_o=dir_3d_line, dir_t=dir_3d_line_homologue)

            # Get color of the point
            point_3d_color = self.images[self.camera_origin][y, x][::-1].astype(np.float32)
            point_3d_result = HAL.project3DScene(point_3d_result)

            # Add color to the 3D point
            point_3d_final = np.append(point_3d_result, point_3d_color)

            if self.verbose:
                print(f"Point 3D: {point_3d_final}")

            # Show 3D point
            GUI.ShowNewPoints([point_3d_final.tolist()])

    def _get_points_interest(self, camera: str) -> np.ndarray:
        """
        Get points of interest in the images applying the Canny algorithm
        :return: Array of points of interest in the left and right image
        """
        edges = cv2.Canny(cv2.cvtColor(self.images[camera], cv2.COLOR_BGR2GRAY), 100, 200)

        GUI.showImages(self.images[camera], edges, True)

        return np.argwhere(edges == 255)

    def _get_3d_point(self, camera: str, x: float, y: float) -> np.ndarray:
        """
        Get 3D point from 2D point
        :param camera: camera (left or right)
        :param x: x coordinate in the image
        :param y: y coordinate in the image
        :return: 3D point
        """
        # Transform the Image Coordinate System to the Camera System
        point_2d = HAL.graficToOptical(camera, [x, y, 1])
        # Back project a 3D Point Space into the 2D Image Point
        point_3d = HAL.backproject(camera, point_2d)

        return point_3d

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
        """
        Get the points of the line that cross the image
        :param p1: Coordinates of the first point
        :param p2: Coordinates of the second point
        :return: Points of the line
        """
        m = (p2[1] - p1[1]) / (p2[0] - p1[0]) + 1e-6
        b = p1[1] - m * p1[0]

        y_final = m * self.w + b
        x_final = (self.h - b) / m

        return (0, int(m * 0 + b)), (int(x_final), self.h) if x_final <= self.w else (self.w, int(y_final))

    def _get_point_homologue(self, cam_o: str, cam_t: str, mask: np.ndarray, point_2d, window_size: int) -> (int, int):
        """
        Get the homologue point of the point_2d in the image cam_t
        :param cam_o: Name of the camera origin
        :param cam_t: Name of the camera target
        :param mask: Mask of the epipolar line
        :param point_2d: Coordinates of the point
        :param window_size: Size of the window
        :return: Coordinates of the homologue point calculated
        """
        # Generate the patch image
        half_w = window_size // 2
        x_min = max(0, point_2d[0] - half_w)
        y_min = max(0, point_2d[1] - half_w)
        x_max = min(self.w, point_2d[0] + half_w + 1)
        y_max = min(self.h, point_2d[1] + half_w + 1)

        template = self.images_hsv[cam_o][y_min:y_max, x_min:x_max]

        mask_img = cv2.bitwise_and(self.images_hsv[cam_t], self.images_hsv[cam_t], mask)

        result = cv2.matchTemplate(mask_img, template, cv2.TM_CCOEFF_NORMED)

        y_result, x_result = np.unravel_index(np.argmax(result), result.shape[:2])

        return x_result + half_w, y_result + half_w

    def _get_triangulation(self, cam_o: str, cam_t: str, dir_o, dir_t) -> np.ndarray:
        """
        Get the 3D point from the triangulation of the 2D points
        :param cam_o: Name of the camera origin
        :param cam_t: Name of the camera target
        :param dir_o: Direction of the 3D line in the camera origin
        :param dir_t: Direction of the 3D line in the camera target
        :return: 3D point
        """
        n = np.cross(dir_o[:3], dir_t[:3])

        A = np.array([dir_o[:3], n, -dir_t[:3]]).T
        b = self.pos_3d_cam[cam_t][:3] - self.pos_3d_cam[cam_o][:3]

        x, r, _ = np.linalg.lstsq(A, b, rcond=None)[0]

        return (x * dir_o[:3]) + ((r / 2) * n)


while True:
    reconstruction = Reconstruction(camera_origin='left', camera_target='right',
                                    image_left=HAL.getImage('left'), image_right=HAL.getImage('right'),
                                    verbose=False)

    reconstruction.algorithm()

    print("Finish the reconstruction")

    while True:
        pass
