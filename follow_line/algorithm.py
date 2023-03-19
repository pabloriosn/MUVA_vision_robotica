from GUI import GUI
from HAL import HAL

import cv2
import math


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float) -> None:
        # Set the proportional, integral and derivative constants
        self._kp = kp
        self._ki = ki
        self._kd = kd

        # Initialize the previous error
        self._previous_error = 0

        # Initialize the integral error
        self._integral = 0

    def control(self, error: float) -> float:
        # Calculate the proportional term
        p = self._kp * error

        # Calculate the integral term
        self._integral += error
        i = self._ki * self._integral

        # Calculate the derivative term
        d = self._kd * (error - self._previous_error)
        self._previous_error = error

        return p + i + d


class CalculateError:
    def __init__(self, width: int = 640,
                 height: int = 480,
                 offset: int = 10,
                 lower_color=(0, 180, 200),
                 upper_color=(0, 255, 255)):

        # Initialize the width and height of the frame
        self._width = width
        self._height = height

        # Offset variable
        self._offset = offset

        # Calculate the center of the frame and add the offset
        self._center = self._width / 2 + offset

        # Define lower and upper HSV threshold values for color filtering.
        self._lower_color = lower_color
        self._upper_color = upper_color

        # Set the centroid's x-coordinate to the center of the frame
        self._centroid_x = self._center

        self._error = 0

    def _preprocess(self, roi):

        # Convert the region of interest (ROI) to HSV color space and filter with the HSV threshold
        mask = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(mask, self._lower_color, self._upper_color)

        return mask

    def _relocalization(self, mask):
        sign = math.copysign(1, self._error)

        while cv2.countNonZero(mask) < 40:
            print(f"Looking for the red line {sign}")

            # Set the linear speed
            HAL.setV(0)
            # Set the angular velocity
            HAL.setW(sign * 0.5)

            frame = HAL.getImage()

            mask = self._preprocess(frame[243:260, :, :])

            # To view a debug image
            GUI.showImage(frame)

        print("Find red line")

        return mask

    def run(self, roi) -> float:

        mask = self._preprocess(roi)

        # Check if robot see the red line
        if cv2.countNonZero(mask) < 40:
            mask = self._relocalization(mask)

        # FInd the contours and select the largest one
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_cnt = max(contours, key=cv2.contourArea)

        # Calculate the centroid of the largest contour
        M = cv2.moments(max_cnt)
        try:
            self._centroid_x = int(M["m10"] / M["m00"])
        except:
            pass

        # Calculate the error between the center of the frame and the centroid calculated
        self._error = self._center - self._centroid_x
        return self._error


base_v = 6.5

prep = CalculateError(width=640, height=480, offset=8, lower_color=(0, 180, 200), upper_color=(0, 255, 255))

controller_w = PIDController(0.001 * 12, 0.00001, 0.001 * 8)
controller_v = PIDController(0.001 * 3, 0, 0.001 * 10)

while True:
    # Get the image
    frame = HAL.getImage()

    # Calculate the error
    error = prep.run(frame[243:260, :, :])

    w = controller_w.control(error)

    v = controller_v.control(error)
    v = base_v - abs(v)

    print(f"Velocity v: {v}")

    # Set the linear speed
    HAL.setV(v)
    # Set the angular velocity
    HAL.setW(w)

    # To view a debug image
    GUI.showImage(frame)
