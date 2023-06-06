import cv2 as cv
import numpy as np
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh

# Indices of eye landmarks
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [ 469, 470, 471, 472]
L_H_LEFT = [33]
L_H_RIGHT = [133]
R_H_LEFT =[362]
R_H_RIGHT = [263]

BLINK_THRESHOLD = 10  # Threshold for detecting a blink (adjust as needed)
BLINK_FRAMES_THRESHOLD = 4  # Number of consecutive frames needed to detect a blink

def euclidean_distance(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance

def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist/total_distance
    iris_position = ""
    if ratio<=0.42:
        iris_position = "Right"
    elif ratio > 0.42 and ratio <= 0.55:
        iris_position = "Center"
    else:
        iris_position = "Left"
    return iris_position, ratio

cap = cv.VideoCapture(0)

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:
    left_blink_counter = 0  # Total blink counter for left eye
    right_blink_counter = 0  # Total blink counter for right eye

    ret, frame = cap.read()
    if not ret:
        exit()

    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img_h, img_w = frame.shape[:2]
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in
                                results.multi_face_landmarks[0].landmark])

        left_eye_points = mesh_points[LEFT_EYE]
        right_eye_points = mesh_points[RIGHT_EYE]

        prev_left_eye_top = np.min(left_eye_points[:, 1])
        prev_left_eye_bottom = np.max(left_eye_points[:, 1])
        prev_right_eye_top = np.min(right_eye_points[:, 1])
        prev_right_eye_bottom = np.max(right_eye_points[:, 1])

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in
                                    results.multi_face_landmarks[0].landmark])

            pixel_str = ' '.join([f'{p[0]} {p[1]}' for p in mesh_points])
            matrix_str = f'[{pixel_str}]\n'
            #print(matrix_str)

            # cv.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
            # cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0, 255, 0), 1, cv.LINE_AA)
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), l_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, mesh_points[R_H_RIGHT][0], 3, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[R_H_LEFT][0], 3, (0, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[L_H_RIGHT][0], 3, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[L_H_LEFT][0], 3, (0, 255, 255), -1, cv.LINE_AA)
            iris_pos, ratio = iris_position(center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT][0])


            left_eye_points = mesh_points[LEFT_EYE]
            right_eye_points = mesh_points[RIGHT_EYE]

            # Calculate the top and bottom vertical positions of left eye landmarks
            left_eye_top = np.min(left_eye_points[:, 1])
            left_eye_bottom = np.max(left_eye_points[:, 1])

            # Calculate the top and bottom vertical positions of right eye landmarks
            right_eye_top = np.min(right_eye_points[:, 1])
            right_eye_bottom = np.max(right_eye_points[:, 1])

            # Check if left eye blink is detected
            if (prev_left_eye_bottom is not None) and (prev_left_eye_top is not None) and \
                    (prev_left_eye_bottom - prev_left_eye_top > BLINK_THRESHOLD) and \
                    (left_eye_bottom - left_eye_top <= BLINK_THRESHOLD):
                left_blink_counter += 1
                #print("Left Eye Blink Detected!")

            # Check if right eye blink is detected
            if (prev_right_eye_bottom is not None) and (prev_right_eye_top is not None) and \
                    (prev_right_eye_bottom - prev_right_eye_top > BLINK_THRESHOLD) and \
                    (right_eye_bottom - right_eye_top <= BLINK_THRESHOLD):
                right_blink_counter += 1
                print("Eye Blink Detected!")

            prev_left_eye_top = left_eye_top
            prev_left_eye_bottom = left_eye_bottom
            prev_right_eye_top = right_eye_top
            prev_right_eye_bottom = right_eye_bottom

        # Display blink counters
        # cv.putText(frame, f"Left Eye Blinks: {left_blink_counter}", (30, 30), cv.FONT_HERSHEY_PLAIN, 2,
        #            (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(frame, f"Eye Blinks: {right_blink_counter}", (30, 30), cv.FONT_HERSHEY_PLAIN, 2,
                   (0, 255, 0), 2, cv.LINE_AA)

        # print(iris_pos)
        cv.putText(frame, f"Iris Position: {iris_pos} {ratio: .2f}", (30, 60), cv.FONT_HERSHEY_PLAIN, 2,
                   (0, 255, 0), 2, cv.LINE_AA)

        cv.imshow('image', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
