import cv2
import numpy as np
import dlib
from imutils import face_utils

# Initializing the camera
cap = cv2.VideoCapture(0)

# Face and landmark detector intialized
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('eye_predictor.dat')

sleep = 0
active = 0
status = ""
color = (0,0,0)

# Getting the eucledian distance
def get_distance(point_1, point_2):
    return np.linalg.norm(point_1 - point_2)

# Getting the EAR
def eye_blink(p1, p2, p3, p4, p5, p6):
    up = get_distance(p2,p4) + get_distance(p3,p5)
    down = get_distance(p1,p6)
    ratio = up/(down*2.0)
    return ratio

# Looping the camera, gathering and analyzing data.
while True:
    h, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Getting the ROI
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Getting the landmarks koordinates
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Getting the ratio for both left and right eye
        left_ratio = eye_blink(landmarks[0], landmarks[1], landmarks[2], landmarks[5], landmarks[4], landmarks[3])
        right_ratio = eye_blink(landmarks[6], landmarks[7], landmarks[8], landmarks[11], landmarks[10], landmarks[9])

        # Getting the mean EAR
        EAR = (left_ratio + right_ratio) / 2.0

        # Checking whether the driver is below or above threshold set for assessing the person as sleeping or not. 
        if (EAR < 0.16):
            sleep = sleep + 1
            drowsy = 0
            active = 0
            if (sleep > 6):
                status = "VARNING!"
                color = (255,0,0)
        else: 
            drowsy = 0
            sleep = 0
            active = active + 1
            if (active > 6):
                status = "Aktiv!"
                color = (0,255,0)

        # Printing the status and EAR value of the driver in the frame displayed
        cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_PLAIN, 1.2, color, 3)
        cv2.putText(frame, "EAR: {:.3f}".format(EAR), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Showing the landmarks used to do calculations in the frame displayed
        for n in range(0, 12):
            (x,y) = landmarks[n]
            cv2.circle(face_frame, (x,y), 1, (255,255,255), -1)

        # Displaying the frame
        cv2.imshow('Frame', frame)
        cv2.imshow("Result of detector", face_frame)
    
    # Adding the functionality to press esc to close the program 
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()

