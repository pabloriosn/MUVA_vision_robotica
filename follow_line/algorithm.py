from GUI import GUI
from HAL import HAL

import cv2


class PIDController:
    def __init__(self, kp, ki, kd) -> None:
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._previous_error = 0
        self._integral = 0

    def control(self, error) -> float:
        p = self._kp * error

        self._integral += error
        i = self._ki * self._integral

        d = self._kd * (error - self._previous_error)
        self._previous_error = error

        return p + i + d


width = 640
height = 480

offset = 8

center = width / 2 + offset

# HSV threshold
lower_color = (0, 180, 200)
upper_color = (0, 255, 255)

controller_w = PIDController(0.001 * 9, 0.00001, 0.001 * 7)
controller_v = PIDController(0.001 * 10, 0, 0)

while True:

    frame = HAL.getImage()

    frame_aux = cv2.cvtColor(frame[243:260, :, :], cv2.COLOR_BGR2HSV)
    frame_aux = cv2.inRange(frame_aux, lower_color, upper_color)

    contours, _ = cv2.findContours(frame_aux, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_cnt = max(contours, key=cv2.contourArea)

    M = cv2.moments(max_cnt)

    try:
        centroid_x = int(M["m10"] / M["m00"])
    except:
        centroid_x = center

    error = center - centroid_x
    print(f"Error: {error}")

    w = controller_w.control(error)
    print(f"Velocity w: {w}")

    if abs(error) > 15:
        v = 4
    else:
        v = 8

    HAL.setV(v)
    HAL.setW(w)

    GUI.showImage(frame)
