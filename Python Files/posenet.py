import cv2
import mediapipe as mp
from network import classify
import serial
import time

arduino = serial.Serial(port='COM6', baudrate=9600, timeout=.1)

def write(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    #data = arduino.readline()
    #return data

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3,640)#width
cap.set(4,480)#height
cap.set(10,20)#brightness

parts=[0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28]#mediapipe inidces corresponding to posenet indexing
letter='S'
prev = letter
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    List = []
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            if id in parts:
                List.append(cx)
                List.append(cy)
        letter=classify(List)
        if letter!=prev:
            write(letter)
    prev=letter
    cv2.putText(img, letter, (20, 150), cv2.FONT_HERSHEY_PLAIN, 12,
                (0, 0, 0), 10)
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # use a delay of 1 ms and break upon 'q' keyboard press
        break
