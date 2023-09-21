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


#
move_servo_to(0, 5)
time.sleep(2)
print(get_distance())



