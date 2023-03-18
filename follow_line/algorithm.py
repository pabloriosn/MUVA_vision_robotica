from GUI import GUI
from HAL import HAL

import cv2


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
        """

        :param error:
        :return:
        """
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
    def __init__(self, width: int = 640, height: int = 480, offset: int = 10,
                 lower_color=(0, 180, 200), upper_color=(0, 255, 255)):
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

    def run(self, roi) -> float:

        # Convert the region of interest (ROI) to HSV color space and filter with the HSV threshold
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi = cv2.inRange(roi, self._lower_color, self._upper_color)

        # FInd the contours and select the largest one
        contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_cnt = max(contours, key=cv2.contourArea)

        # Calculate the centroid of the largest contour
        M = cv2.moments(max_cnt)
        try:
            self._centroid_x = int(M["m10"] / M["m00"])
        except:
            pass

        # Calculate the error between the center of the frame and the centroid calculated
        return self._center - self._centroid_x


prep = CalculateError(width=640, height=480, offset=8, lower_color=(0, 180, 200), upper_color=(0, 255, 255))

controller_w = PIDController(0.001 * 9, 0.00001, 0.001 * 7)
controller_v = PIDController(0.001 * 10, 0, 0)

while True:

    # Get the image
    frame = HAL.getImage()

    # Calculate the error
    error = prep.run(frame[243:260, :, :])
    print(f"Error: {error}")

    w = controller_w.control(error)
    print(f"Velocity w: {w}")

    if abs(error) > 15:
        v = 4
    else:
        v = 4

    # Set the linear speed
    HAL.setV(v)
    # Set the angular velocity
    HAL.setW(w)

    # To view a debug image
    GUI.showImage(frame)
