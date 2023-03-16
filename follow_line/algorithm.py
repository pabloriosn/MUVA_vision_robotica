from GUI import GUI
from HAL import HAL

import cv2

width = 640
height = 480

center = width / 2

offset = 10

# HSV threshold
lower_color = (0, 180, 200)
upper_color = (0, 255, 255)

kp = 0.001 * 9
kd = 0.001 * 7

last_error = 0
w = 0

while True:

    frame = HAL.getImage()

    frame_aux = cv2.cvtColor(frame[243:260, :, :], cv2.COLOR_BGR2HSV)
    frame_aux = cv2.inRange(frame_aux, lower_color, upper_color)

    M = cv2.moments(frame_aux)

    try:
        centroid_x = int(M["m10"] / M["m00"])
    except:
        centroid_x = center

    error = center - centroid_x

    w = kp * error + kd * (error - last_error)

    last_error = error
    print(f"Error: {error}")

    HAL.setV(4)
    HAL.setW(w)

    # frame = cv2.line(frame, (0, 243), (width, 243), (255, 255 , 255), 1)
    # frame = cv2.line(frame, (0, 260), (width, 260), (255, 255 , 255), 1)

    cv2.circle(frame, (centroid_x, 250), 3, (0, 0, 0), -1)
    frame = cv2.line(frame, (int(width / 2), 0), (int(width / 2), height), (255, 255, 255), 1)

    GUI.showImage(frame)
