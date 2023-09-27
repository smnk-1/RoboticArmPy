from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import time
import sys
import serial
import time

# SERIAL SETUP
comPort = 'COM5'
ser = serial.Serial(comPort, baudrate=9600, timeout=2)

# ARDUINO PIN SETUP
BaseServoPin = 5
LowJointServoPin = 4
HighJointServoPin = 3
HandTitlServoPin = 2
HandUpDownServoPin = 6
HandOpenCloseServoPin = 7

BaseServoCurDeg = 90
LowJointServoCurDeg = 120
HighJointServoCurDeg = 175
HandTitlServoCurDeg = 100
HandUpDownServoCurDeg = 40
HandOpenCloseServoCurDeg = 180

BaseServoCurDegH = 90
LowJointServoCurDegH = 120
HighJointServoCurDegH = 175
HandTitlServoCurDegH = 100
HandUpDownServoCurDegH = 40
HandOpenCloseServoCurDegH = 180


def read_serial():
    try:
        data = ser.readline()
        return int(data[0:len(data)-2].decode('utf-8'))
    except (serial.SerialException, ValueError, AttributeError):
        # print('Error reading or decoding the data from Arduino')
        return None


def get_distance():
    dist = None
    while not isinstance(dist, int):
        msg = 'd;'
        msg = bytes(str(msg), 'utf-8')
        ser.write(msg)
        # print(msg)
        dist = read_serial()
    return dist


def move_servo_to(deg, servonum):
    y = None
    while not isinstance(y, int):
        msg = 's' + str(servonum) + str(deg) + ';'  # s(1-6)(0-180);
        msg = bytes(str(msg), 'utf-8')
        ser.write(msg)
        # print(msg)
        y = read_serial()


def move_base_servo(angle):
    global BaseServoCurDeg
    if 0 <= BaseServoCurDeg + angle <= 180:
        BaseServoCurDeg += angle
        move_servo_to(BaseServoCurDeg, BaseServoPin)
    else:
        if BaseServoCurDeg + angle > 180 and BaseServoCurDeg != 180:
            move_servo_to(180, BaseServoPin)
            BaseServoCurDeg = 180
        elif BaseServoCurDeg + angle < 0 and BaseServoCurDeg != 0:
            move_servo_to(0, BaseServoPin)
            BaseServoCurDeg = 0


def catch():
    global LowJointServoCurDeg, HighJointServoCurDeg, LowJointServoPin, HighJointServoPin, HandOpenCloseServoCurDeg, HandOpenCloseServoPin, BaseServoCurDeg
    if get_distance() <= 35:
        while True:
            x = get_distance()
            if x > 4.5:
                move_servo_to(LowJointServoCurDeg - 5, LowJointServoPin)
                LowJointServoCurDeg -= 5
                if HighJointServoCurDeg >= 35:
                    move_servo_to(HighJointServoCurDeg - 5, HighJointServoPin)
                    HighJointServoCurDeg -= 5
                    time.sleep(0.3)
            elif x <= 4.5:
                y = get_distance()
                if y <= 4.5:
                    move_servo_to(90, HandOpenCloseServoPin)
                    HandOpenCloseServoCurDeg = 90
                    time.sleep(0.5)
                    break

            if HighJointServoCurDeg < 35 and LowJointServoCurDeg <= 0:
                break

        # Go home
        move_servo_to(LowJointServoCurDegH, LowJointServoPin)
        LowJointServoCurDeg = LowJointServoCurDegH
        move_servo_to(HighJointServoCurDegH, HighJointServoPin)
        HighJointServoCurDeg = HighJointServoCurDegH
        time.sleep(1)
        move_servo_to(0, BaseServoPin)
        BaseServoCurDeg = 0
        time.sleep(2.5)
        move_servo_to(180, HandOpenCloseServoPin)
        HandOpenCloseServoCurDeg = 180
    else:
        pass


rotating_direction = 1

# WEBCAM
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(2, 720)

leftline = [590+80, 0, 590+80, 1280]
rightline = [690+80, 0, 690+80, 1280]
leftline2 = [400+80, 0, 400+80, 1280]
rightline2 = [880+80, 0, 880+80, 1280]

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

    if len(IDsList) > 0:
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
                    move_base_servo(0.5)
                elif cx <= leftline2[0]:
                    move_base_servo(3)
                elif rightline2[0] > cx > rightline[0]:
                    move_base_servo(-0.5)
                elif cx > rightline2[0]:
                    move_base_servo(-3)
                elif leftline[0] < cx < rightline[0]:
                    print(get_distance())
                    catch()

    else:
        if 0 < BaseServoCurDeg < 180:
            move_base_servo(rotating_direction)
        elif BaseServoCurDeg == 0:
            rotating_direction = 1
            move_base_servo(rotating_direction)
        elif BaseServoCurDeg == 180:
            rotating_direction = -1
            move_base_servo(rotating_direction)
        # print(BaseServoCurDeg)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)




