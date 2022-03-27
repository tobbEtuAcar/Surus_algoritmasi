#!/usr/bin/env python
# -*-coding: utf-8 -*-

from copy import copy
from tkinter import E
import cv2
import numpy as np
import rospy
import time
import torch
import LineDet
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image

laser_data = list()
flag = 0
counter = 0
MOD = 'SAG REFERANS'
MOD_PREV = 'SAG REFERANS'
DURAKModuBitisZamani = 0.0
start_timer = 0
stop_counter = 0
stop_flag = 0

model = torch.hub.load('/home/feanor/yolov5', 'custom', path='/home/feanor/yolov5/best.pt', source='local')
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
bridge = CvBridge()

# global speed
# global steering_angle	
steering_angle = 0.0
speed = 1.3


def lidar_callback(lidar_data):
    # print('callback')
    global laser_data
    laser_data = lidar_data
    laser_data = np.array(laser_data.ranges)
    laser_data[laser_data > 3.5] = 3.5
    regions = {
        'right': sum(laser_data[155:310]) / 155,
        'front_right': sum(laser_data[310:464]) / 155,
        'front': sum(laser_data[465:619]) / 155,
        'front_left': sum(laser_data[620:774]) / 155,
        'left': sum(laser_data[770:925]) / 155
    }
    # movement(regions)


def movement(angle, direction):
    global flag
    global counter
    global speed
    global steering_angle
    global start_timer
    global MOD
    global MOD_PREV
    global stop_flag, stop_counter

    if MOD == 'SAG REFERANS':

        print("angle: "+str(angle))
        print("flag: "+str(flag))
        if direction == 'DUZ' and flag == 0:
            counter = 0
            if angle >= 178 or angle <= 2:
                speed = 1.3
                steering_angle = 0
                # print('duz')

            elif angle < 178 and angle > 90:
                speed = 1.3
                steering_angle = 0.025
                # print('sola')

            elif angle > 2 and angle < 90:
                speed = 1.3
                steering_angle = -0.025
                # print('sağa')
        else:
            speed = 1
            counter += 1
            if counter == 3:
                flag = 1

            if counter > 68 and counter < 73:
                steering_angle = 0
            elif counter >= 73:
                flag = 0
            elif counter > 17:
                steering_angle = -0.6

            # print(counter)

        """if regions['right'] > 0.3:
            speed = 2
            
            if regions['right'] > 0.99000 and regions['right'] < 1.05:  
                # print('DÜZ GİDECEN')
                steering_angle = 0.0
                speed = 2 
                flag = 0
                counter = 0

            elif regions['right'] > 2.1:   
                # print('HARD RIGHT')
                steering_angle = -0.4
                speed = 1.8
                flag = 1

            elif regions['right'] > 1.05 and regions['right'] < 2.1 and flag == 1: 
                # print('HARD RIGHT')
                steering_angle = -0.4
                speed = 1.8

            elif  regions['right'] > 1.05 and flag == 0: 
                # print('SOFT RIGHT')  
                steering_angle = -0.02
                speed = 1.8   

            elif regions['right'] < 0.99000 and regions['left'] > 1.2:  
                # print('HARD LEFT')
                steering_angle = 0.5
                speed = 1.8


            elif regions['right'] < 0.99000 and regions['right'] > 0.9  and flag == 0:   
                # print('SOFT LEFT')
                steering_angle = 0.02
                speed = 1.8
            
        else: 
            # print('STOP!')
            speed = 0.0
            steering_angle = 0.0
    """
    elif MOD == 'SOL REFERANS':
        print()

    elif MOD == 'DURAK':
        stop_counter += 1
        print(stop_counter)
        print(stop_flag)
        print(speed)
        if stop_counter < 160:
            if direction == 'DUZ' and flag == 0:
                counter = 0
                if angle >= 178 or angle <= 2:
                    speed = 1.3
                    steering_angle = 0
                    # print('duz')
                elif angle < 178 and angle > 90:
                    speed = 1.3
                    steering_angle = 0.025
                    # print('sola')

                elif angle > 2 and angle < 90:
                    speed = 1.3
                    steering_angle = -0.025
                    # print('sağa')
        if stop_counter >= 160 and stop_flag < 250:
            speed = 0
            stop_flag += 1
            if stop_flag >= 250:
                stop_counter = 0
                MOD = 'SOL REFERANS'


    obj.drive.speed = speed
    obj.drive.steering_angle = steering_angle
    pub.publish(obj)


def lineDetectionCallback(mesaj):
    frame = bridge.imgmsg_to_cv2(mesaj, "bgr8")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_copy = np.copy(gray)
    show = np.copy(frame)
    copy_frame = np.copy(frame)
    Roi = LineDet.region_of_interest(gray)
    kernel = 9
    blur = cv2.GaussianBlur(Roi, (kernel, kernel), 0)
    copy_Roi = np.copy(blur)
    endpoint_coords = LineDet.get_coordinate(copy_Roi)
    for i in range(0, len(endpoint_coords)):
        cv2.circle(copy_frame, (endpoint_coords[i][0].astype(int),
                                endpoint_coords[i][1].astype(int)), 5, (0, 0, 255),
                   cv2.FILLED)
    duzDonusState = LineDet.state_control(endpoint_coords)
    left_coords, right_coords = LineDet.left_right_coordinates(endpoint_coords)
    print(endpoint_coords)
    print(right_coords)
    print(left_coords)
    if len(left_coords) and len(right_coords) >= 2:
        mid_coords = LineDet.find_mid_coordinates(left_coords, right_coords)
        # for i in range(len(mid_coords)):
        #     cv2.circle(copy_frame, (mid_coords[i][0].astype(int),
        #                             mid_coords[i][1].astype(int)), 5, (0, 0, 255), cv2.FILLED)
        cam_line = LineDet.find_camera_line(mid_coords, gray_copy)
        show = LineDet.draw_lines(cam_line, mid_coords, show)
        angle, direction = LineDet.find_angle_info(mid_coords)
        # print('Angle: ', angle)
        # print('Direction: ', duzDonusState)
        movement(angle, duzDonusState)
    else:
        movement(-1, 'DON')

    cv2.imshow("Linedet", copy_frame)
    cv2.imshow("Linedet2", show)
    cv2.imshow("Roi", Roi)
    cv2.waitKey(1)


def cameraCallback(mesaj):
    foto = bridge.imgmsg_to_cv2(mesaj, "bgr8")
    results = model(foto)

    global MOD
    global MOD_PREV
    global DURAKModuBitisZamani

    print('MOD:', MOD)
    print('****************')

    for *box, conf, cls in results.pred[0]:
        # print(results.pandas().xyxy[0])
        #print(results.pandas().xyxy[0])
        sign_num = len(results.pandas().xyxy[0])
        for i in range(sign_num):
            label = results.pandas().xyxy[0].name[i]
            raw_conf = results.pandas().xyxy[0].confidence[i]
            conf = float(f'{raw_conf:.2}')
            if label == 'durak' and conf >= 0.95 and MOD != 'DURAK':
                print(conf)
                MOD_PREV = MOD
                MOD = 'DURAK'
                break

            elif label == 'saga_don' and conf >= 0.93 and MOD != 'SAG REFERANS':
                MOD_PREV = MOD
                MOD = 'SAG REFERANS'
                break
            elif label == 'sola_don' and conf >= 0.93 and MOD != 'SOL REFERANS':
                MOD_PREV = MOD
                MOD = 'SOL REFERANS'
                break

        # if results.names[int(cls)].find('durak') != -1 and float(f'{conf:.2f}') > 0.85 and MOD != 'DURAK':
        #     MOD_PREV = MOD
        #     MOD = 'DURAK'
        #     break
        #     # print(f'{conf:.2f}')
        # elif results.names[int(cls)].find('saga_don') != -1 and float(f'{conf:.2f}') > 0.6 and MOD != 'SAG REFERANS':
        #     MOD_PREV = MOD
        #     MOD = 'SAG REFERANS'
        #     break
        #
        # elif results.names[int(cls)].find('sola_don') != -1 and float(f'{conf:.2f}') > 0.4 and MOD != 'SOL REFERANS':
        #     MOD_PREV = MOD
        #     MOD = 'SOL REFERANS'
        #     break

        # label = f'{results.names[int(cls)]} {conf:.2f}'
        # c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

    cv2.imshow("Arac Kamerasi", results.render()[0])
    # cv2.imshow("Arac Kamerasi", foto)
    cv2.waitKey(1)


if __name__ == '__main__':
    rospy.init_node('drive', anonymous=True)
    rospy.Subscriber('/scan', LaserScan, lidar_callback)
    rospy.Subscriber("/camera/zed/rgb/image_rect_color", Image, cameraCallback)
    rospy.Subscriber("/camera/zed/rgb/image_rect_color", Image, lineDetectionCallback)

    rospy.loginfo("Press CTRL + C for stopping the simulation")

    pub = rospy.Publisher("/vesc/high_level/ackermann_cmd_mux/output", AckermannDriveStamped, queue_size=10)

    obj = AckermannDriveStamped()

    rospy.spin()
