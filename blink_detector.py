import cv2
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
    cv2.line(frame, (left_end.x, left_end.y), (right_end.x, right_end.y), (255, 0, 0))
    cv2.line(frame, center_botton, center_top, (255, 255, 0))

    # Calculating lengths of line
    hor_line_length = hypot((left_end.x - right_end.x), (left_end.y - right_end.y))
    vert_line_length = hypot((center_top[0] - center_botton[0]), (center_top[1] - center_botton[1]))
    
    ratio = hor_line_length/vert_line_length

    return ratio

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
            cv2.putText(frame, "BLINKING", (50, 150), font, 2, (0, 0, 0))


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()