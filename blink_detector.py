import cv2
import winsound
import numpy as np 
import dlib
from math import hypot

font = cv2.FONT_HERSHEY_PLAIN
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


def get_blink_ratio(eye_points, facial_landmark):
    # Locating Poinits an eyes
    left_end = facial_landmark.part(eye_points[0])
    right_end = facial_landmark.part(eye_points[3])
    # Center ponits using midpoint function
    center_top = midpoint(facial_landmark.part(eye_points[1]), facial_landmark.part(eye_points[2]))
    center_botton = midpoint(facial_landmark.part(eye_points[5]), facial_landmark.part(eye_points[4]))

    # Drawing lines
    # cv2.line(frame, (left_end.x, left_end.y), (right_end.x, right_end.y), (255, 0, 0))
    # cv2.line(frame, center_botton, center_top, (255, 255, 0))

    # Calculating lengths of line
    hor_line_length = hypot((left_end.x - right_end.x), (left_end.y - right_end.y))
    vert_line_length = hypot((center_top[0] - center_botton[0]), (center_top[1] - center_botton[1]))
    
    ratio = hor_line_length/vert_line_length

    return ratio


def get_gaze_ratio(eye_points, facial_landmark):
    left_eye_region = np.array([(facial_landmark.part(eye_points[0]).x, facial_landmark.part(eye_points[0]).y),
                                (facial_landmark.part(eye_points[1]).x, facial_landmark.part(eye_points[1]).y),
                                (facial_landmark.part(eye_points[2]).x, facial_landmark.part(eye_points[2]).y),
                                (facial_landmark.part(eye_points[3]).x, facial_landmark.part(eye_points[3]).y),
                                (facial_landmark.part(eye_points[4]).x, facial_landmark.part(eye_points[4]).y),
                                (facial_landmark.part(eye_points[5]).x, facial_landmark.part(eye_points[5]).y)], np.int32)

    # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255))


    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)

    cv2.polylines(mask, [left_eye_region], True, 255)
    cv2.fillPoly(mask, [left_eye_region], 255)

    eye = cv2.bitwise_and(gray, gray, mask=mask)

    # locating extreme ponits of the eyes
    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    grey_eye = eye[min_y:max_y, min_x:max_x]
    _, threshold_eye = cv2.threshold(grey_eye, 127, 255, cv2.THRESH_BINARY)

    threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
    cv2.imshow("Threshold", threshold_eye)

    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0 : height, 0 : int(width/2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0 : height, int(width/2) : width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    gaze_ratio = left_side_white/right_side_white if right_side_white > 0 else 0

    return gaze_ratio


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        # Print face coordinates and box around
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0))

        landmarks = predictor(gray, face)

        ### BLINK DETECTION ###

        # Print landmark coordinated and circle around
        # part1 = landmarks.part(37)
        # part2 = landmarks.part(38)
        # part3 = landmarks.part(40)
        # part4 = landmarks.part(41)
        # cv2.circle(frame, (part1.x, part1.y), 3, (0, 255, 255), 0)
        # cv2.circle(frame, (part2.x, part2.y), 3, (0, 255, 255), 0)
        # cv2.circle(frame, (part3.x, part3.y), 3, (0, 255, 255), 0)
        # cv2.circle(frame, (part4.x, part4.y), 3, (0, 255, 255), 0)

        left_ratio = get_blink_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_ratio = get_blink_ratio([42, 43, 44, 45, 46, 47], landmarks)

        blink_ratio = (left_ratio + right_ratio)/2

        if blink_ratio > 5:
            cv2.putText(frame, "BLINKING", (50, 200), font, 2, (0, 0, 0))
            frequency = 1500
            duration = 100
            winsound.Beep(frequency, duration)

        # ### GAZE DETECTION ###

        gaze_ratio_left = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)

        gaze_ratio = (gaze_ratio_left + gaze_ratio_right)/2
        cv2.putText(frame, str(gaze_ratio), (50, 200), font, 2, (0,0,0))
        if gaze_ratio <= 0.8:
            cv2.putText(frame, "RIGHT GAZING", (50, 100), font, 2, (0,0,0))
        elif 1 < gaze_ratio < 1.8:
            cv2.putText(frame, "CENTER GAZING", (50, 100), font, 2, (0,0,0))
        else:
            cv2.putText(frame, "LEFT GAZING", (50, 100), font, 2, (0,0,0))
      

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()