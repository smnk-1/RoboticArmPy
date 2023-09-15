from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import pyfirmata
from time import sleep

# PYFIRMATA SETUP
ComPort = 'COM5'
Board = pyfirmata.Arduino(ComPort)

# SERVO HOME POSISIONS

Board.digital[5].mode = pyfirmata.SERVO
Board.digital[5].write(90)
BaseServoCurrentAngle = 90

Board.digital[4].mode = pyfirmata.SERVO
Board.digital[4].write(120)
LowJointServoCurrentAngle = 120

Board.digital[3].mode = pyfirmata.SERVO
Board.digital[3].write(175)
HighJointServoCurrentAngle = 175

Board.digital[2].mode = pyfirmata.SERVO
Board.digital[2].write(107)
HandTiltServoCurrentAngle = 107

Board.digital[6].mode = pyfirmata.SERVO
Board.digital[6].write(50)
HandUpDownServoCurrentAngle = 50

Board.digital[7].mode = pyfirmata.SERVO
Board.digital[7].write(180)
HandOpenCloseServoCurrentAngle = 180


def move_base_servo(angle):
    global BaseServoCurrentAngle
    if 0 <= BaseServoCurrentAngle + angle <= 180:
        BaseServoCurrentAngle += angle
        Board.digital[5].write(BaseServoCurrentAngle)
    else:
        pass


# WEBCAM
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(2, 720)

leftline = [580, 0, 580, 1280]
rightline = [700, 0, 700, 1280]
leftline2 = [400, 0, 400, 1280]
rightline2 = [880, 0, 880, 1280]

model = YOLO('../YoloWeights/yolov8l.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

while True:
    success, img = cap.read()

    results = model(img, stream=True)
    # img = cv2.rotate(img, cv2.ROTATE_180)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil(box.conf[0]*100)/100

            # Class name
            cls = box.cls[0]

            currenClass = classNames[int(cls)]

            if currenClass == 'bottle':
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTraker = tracker.update(detections)

    cv2.line(img, (leftline[0], leftline[1]), (leftline[2], leftline[3]), (255, 0, 0), 3)   # blue
    cv2.line(img, (rightline[0], rightline[1]), (rightline[2], rightline[3]), (0, 0, 255), 3)   # red
    cv2.line(img, (leftline2[0], leftline2[1]), (leftline2[2], leftline2[3]), (0, 255, 255), 3)
    cv2.line(img, (rightline2[0], rightline2[1]), (rightline2[2], rightline2[3]), (0, 255, 255), 3)

    IDsList = []
    for result in resultsTraker:
        x1, y1, x2, y2, ID = result
        IDsList.append(ID)

    for result in resultsTraker:
        x1, y1, x2, y2, ID = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=2, colorR=(255, 255, 0))
        cvzone.putTextRect(img, f'{int(ID)}', (max(0, x1), max(35, y1 - 20)), scale=1, offset=3, thickness=1)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        if ID == min(IDsList):
            if leftline2[0] < cx <= leftline[0]:
                move_base_servo(1)
            elif cx <= leftline2[0]:
                move_base_servo(4)
            elif rightline2[0] > cx > rightline[0]:
                move_base_servo(-1)
            elif cx > rightline2[0]:
                move_base_servo(-4)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)
