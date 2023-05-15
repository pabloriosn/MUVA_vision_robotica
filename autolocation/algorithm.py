import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_camera_matrix(points3d, points2d, camera_matrix, dist_coeff):

    retval, rvec, tvec = cv2.solvePnP(points3d, points2d, camera_matrix, dist_coeff, flags=cv2.SOLVEPNP_IPPE)

    return (retval, rvec, tvec) if retval else None


def draw_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim3d([0, 100])
    ax.set_ylim3d([0, 100])
    ax.set_zlim3d([0, 100])

    # Define las coordenadas de los vértices del cuadrado
    x = [0, 0, 9.5, 9.5, 0]
    y = [0, 9.5, 9.5, 0, 0]
    z = [0, 0, 0, 0, 0]

    # Utiliza la función plot para dibujar el cuadrado
    ax.plot(x, y, z, 'b')

    plt.show(block=False)

    return ax


def draw_point(ax, rvec, tvec):
    R = cv2.Rodrigues(rvec)

    point = (-R[0].T @ tvec) / 10
    point_x = (R[0].T @ (np.array([[30, 0, 0]]).T - tvec)) / 10
    point_y = (R[0].T @ (np.array([[0, 30, 0]]).T - tvec)) / 10
    point_z = (R[0].T @ (np.array([[0, 0, 60]]).T - tvec)) / 10

    # Plot camera position
    # ax.scatter3D(*point.astype(int), c='y', marker='x')

    # Plot axes of the camera
    ax.plot([point[0], point_x[0]], [point[1], point_x[1]], [point[2], point_x[2]], 'r')
    ax.plot([point[0], point_y[0]], [point[1], point_y[1]], [point[2], point_y[2]], 'g')
    ax.plot([point[0], point_z[0]], [point[1], point_z[1]], [point[2], point_z[2]], 'b')

    plt.draw()
    plt.pause(0.1)


def main():
    # Load camera matrix and distortion coefficients
    matrix_camera = np.load('matrix_calibration.npz')
    camera_matrix = matrix_camera['arr_0']
    dist_coeffs = matrix_camera['arr_1']

    # Real world coordinates of the corners of the aruco marker
    points3d = np.array([[0, 0, 0], [0, 95, 0], [95, 95, 0], [95, 0, 0]], dtype=np.float32)

    # Create instance of VideoCapture
    cam = cv2.VideoCapture(2)

    # Create dictionary of aruco markers
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_100)
    aruco_parameters = cv2.aruco.DetectorParameters_create()

    # Draw plot
    ax = draw_plot()

    while cam.isOpened():
        # Read frame from camera
        ret, frame = cam.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_parameters)

            if ids is not None:
                frame = cv2.aruco.drawDetectedMarkers(frame, corners, borderColor=(0, 0, 255))

                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.1, camera_matrix, dist_coeffs)

                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs, tvecs, 0.01)

                retval, rvec, tvec = get_camera_matrix(points3d, corners[0], camera_matrix, dist_coeffs)

                if retval:
                    draw_point(ax, rvec, tvec)

            cv2.imshow('Camara', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
