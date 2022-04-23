#!/usr/bin/env python
# -*-coding: utf-8 -*-

from copy import copy
from itertools import count
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
reset = 0
counter = 0
counterPark = 0
MOD = 'SAG REFERANS'
MOD_PREV = 'SAG REFERANS'
start_timer = 0
stop_counter = 0
stop_flag = 0

model = torch.hub.load('/home/talhaunal/yolov5', 'custom', path='/home/talhaunal/yolov5/best.pt', source='local')
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
bridge = CvBridge()

# global speed
# global steering_angle	
steering_angle = 0.0
speed = 1.3


def lidar_callback(lidar_data):
    # # print('callback')
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
    global reset
    global MOD_PREV
    global stop_flag, stop_counter
    global counterPark


    if MOD == 'PARK':
        counterPark += 1
    # # print("PARK COUNTER",counterPark)

    if MOD == 'SAG REFERANS':

        if reset == 1:
            reset = 0
            flag = 0
            counter = 0

        ## print(angle)
        ## print('counter')
        ## print(counter)
        if direction == 'DUZ' and flag == 0:
            counter = 0

            if angle >= 178 or angle <= 2:
                speed = 1.3
                steering_angle = 0
                # print('duz')
            
            elif angle < 178 and angle > 90:
                speed = 1.3
                angleBolum = 178 - angle
                # print('sola')
                steering_angle = 0.005 * angleBolum             

            elif angle > 2 and angle < 90: 
                speed = 1.3
                angleBolum = 178 - (180- angle)
                # print('sağa')
                steering_angle = -0.005 * angleBolum
                
        elif direction == 'KESKIN DON' and flag != 1:   
            speed = 1
            counter += 1
            if counter == 3:
                flag = 2

            if counter > 45 and counter < 55:
                steering_angle = 0
            elif counter >= 55:
                flag = 0
            elif counter > 18: #ne kadar geç döneceğine bakıyor
                steering_angle = -0.60

        elif flag != 2:
            speed = 1
            counter += 1
            if counter == 3:
                flag = 1

            if counter > 68 and counter < 77:
                steering_angle = 0
            elif counter >= 77:
                flag = 0
            elif counter > 18: #ne kadar geç döneceğine bakıyor
                steering_angle = -0.60

    elif MOD == 'SOL REFERANS':
        if reset == 0:
            reset = 1
            flag = 0
            counter = 0

        if direction == 'DUZ' and flag == 0:
            counter = 0

            if angle >= 178 or angle <= 2:
                speed = 1.3
                steering_angle = 0
                # print('duz')
            
            elif angle < 178 and angle > 90:
                speed = 1.3
                angleBolum = 178 - angle
                # print('sola')
                steering_angle = 0.005 * angleBolum
                
            elif angle > 2 and angle < 90: 
                speed = 1.3
                angleBolum = 178 - (180- angle)
                # print('sağa')
                steering_angle = -0.005 * angleBolum
        elif (direction == 'KESKIN DON' and flag != 1 and flag != 3) or flag == 2:   
            speed = 1
            counter += 1
            if counter == 3:
                flag = 2

            if counter > 92 and counter < 98:
                steering_angle = 0
            elif counter >= 98:
                flag = 0
            elif counter > 45: #ne kadar geç döneceğine bakıyor
                steering_angle = 0.60
        elif (direction == 'T DONUS') or flag == 3:
            speed = 1
            counter += 1
            if counter == 2:
                flag = 3

            if counter > 60 and counter < 67:
                steering_angle = 0
            elif counter >= 67:
                flag = 0
            elif counter > 13: #ne kadar geç döneceğine bakıyor
                steering_angle = 0.60
        elif  flag != 2 and flag != 3:
            speed = 1
            counter += 1
            if counter == 2:
                flag = 1

            if counter > 60 and counter < 67:
                steering_angle = 0
            elif counter >= 67:
                flag = 0
            elif counter > 13: #ne kadar geç döneceğine bakıyor
                steering_angle = 0.60

    elif MOD == 'DURAK':
        flag = 0
        counter = 0
        if angle >= 178 or angle <= 2:
            speed = 1.3
            steering_angle = 0
            # print('duz')
        
        elif angle < 178 and angle > 90:
            speed = 1.3
            angleBolum = 178 - angle
            # print('sola')
            steering_angle = 0.005 * angleBolum
            
        elif angle > 2 and angle < 90: 
            speed = 1.3
            angleBolum = 178 - (180- angle)
            # print('sağa')
            steering_angle = -0.005 * angleBolum

    elif MOD == 'DURAK_2':
        time.sleep(1)
        speed = 0.0
        time.sleep(10)    
        MOD = 'SOL REFERANS'

    elif MOD == 'DEVAM':
        flag = 0
        if angle >= 178 or angle <= 2:
            speed = 1.3
            steering_angle = 0
            # print('duz')
            
        elif angle < 178 and angle > 90:
        # stop_counter = 0    
            speed = 1.3
            angleBolum = 178 - angle
            # print('sola')
            steering_angle = 0.005 * angleBolum
                
        elif angle > 2 and angle < 90: 
            speed = 1.3
            angleBolum = 178 - (180- angle)
            # print('sağa')
            steering_angle = -0.005 * angleBolum
    
    elif MOD == 'PARK':
        if(counterPark < 155):
            flag = 0
            if angle >= 178 or angle <= 2:
                speed = 1.3
                steering_angle = 0
                # print('duz')
                
            elif angle < 178 and angle > 90:
                speed = 1.3
                angleBolum = 178 - angle
                # print('sola')
                steering_angle = 0.005 * angleBolum
                    
            elif angle > 2 and angle < 90: 
                speed = 1.3
                angleBolum = 178 - (180- angle)
                # print('sağa')
                steering_angle = -0.005 * angleBolum
        elif counterPark == 175:
            steering_angle = 0.0
        elif (counterPark < 175):
            steering_angle += 0.1
        elif (counterPark < 195):
            steering_angle -= 0.1
        else: 
            steering_angle = 0
    print(MOD)
    print('*******')
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
    # # print('Left Koord: ', len(left_coords))
    # # print('Right Koord: ', len(right_coords))

    if len(left_coords) == 4 and len(right_coords) == 4:
        # print('Direction: DON')
        movement(-1,'DON')
    elif len(left_coords) == 0 and len(right_coords) == 0:
        # print('Direction: T DONUS')
        movement(-1,'T DONUS')
    elif len(left_coords) == 2 and len(right_coords) == 2:
        mid_coords = LineDet.find_mid_coordinates(left_coords, right_coords)
        # for i in range(len(mid_coords)):
        #     cv2.circle(copy_fraprint("Label: ", label, "Conf:", conf)me, (mid_coords[i][0].astype(int),
        #                             mid_coords[i][1].astype(int)), 5, (0, 0, 255), cv2.FILLED)
        cam_line = LineDet.find_camera_line(mid_coords, gray_copy)
        show = LineDet.draw_lines(cam_line, mid_coords, show)
        angle, direction = LineDet.find_angle_info(mid_coords)
        # # print('Angle: ', angle)
        # # print('Direction: ', duzDonusState)
        movement(angle, duzDonusState)
    elif (len(left_coords) == 1 and len(right_coords) == 1) or (len(left_coords) == 3 and len(right_coords) == 0):
        # # print('Direction: KESKIN')
        movement(-1,'KESKIN DON')
    else:
        # print('Direction: DON')
        movement(-1,'DON')

    # cv2.imshow("Skeleton Endpoints", copy_frame)
    # cv2.imshow("Skeleton Lines", show)
    # cv2.imshow("Roi", Roi)
    cv2.waitKey(1)


def cameraCallback(mesaj):
    foto = bridge.imgmsg_to_cv2(mesaj, "bgr8")
    results = model(foto)

    global MOD
    global MOD_PREV
    global counterPark

    # print('MOD:', MOD)
    # print('****************')

    # if(len(results.pred[0]) == 0 and MOD == 'DEVAM' and counterDevam > 665):
    #     MOD = 'SOL REFERANS'

    sign_num = len(results.pandas().xyxy[0])
    print('tabela sayisi:', sign_num)
    for i in range(sign_num):
        x_max = results.pandas().xyxy[0].xmax[i]
        x_min = results.pandas().xyxy[0].xmin[i]
        y_max = results.pandas().xyxy[0].ymax[i]
        y_min = results.pandas().xyxy[0].ymin[i]
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        bbox_area = int(bbox_width * bbox_height)
        print('Bounding Box Area: ', bbox_area)
        print('X_min: ', x_min)
        print('X_max: ', x_max)
        
        label = results.pandas().xyxy[0].name[i]
        raw_conf = results.pandas().xyxy[0].confidence[i]
        conf = float(f'{raw_conf:.2}')
        print(conf)
        if label == 'durak' and conf >= 0.95 and MOD != 'DURAK':
            MOD_PREV = MOD
            MOD = 'DURAK'
            break
        elif label == 'durak' and x_max > 636 and x_min > 550 and MOD == 'DURAK':
            MOD_PREV = MOD
            MOD = 'DURAK_2'
            break
        elif bbox_area < 800 or x_min < 3 or x_max > 638:
            print("Ignored Label:", label)
            continue
        elif label == 'sola_don' and conf >= 0.80 and MOD != 'SOL REFERANS':
            MOD_PREV = MOD
            MOD = 'SOL REFERANS'
            break
        elif (label == 'sola_donulmez' or label == 'saga_donulmez') and conf >= 0.93 and MOD != 'DEVAM':
            MOD_PREV = MOD
            MOD = 'DEVAM'
            break
        elif label == 'saga_don' and conf >= 0.93 and MOD != 'SAG REFERANS':
            MOD_PREV = MOD
            MOD = 'SAG REFERANS'
            break
        elif label == 'park'  and conf >= 0.93 and MOD != 'PARK':
            MOD_PREV = MOD
            MOD = 'PARK'
            break

        print('****************')
        # if results.names[int(cls)].find('durak') != -1 and float(f'{conf:.2f}') > 0.85 and MOD != 'DURAK':
        #     MOD_PREV = MOD
        #     MOD = 'DURAK'
        #     break
        #     # # print(f'{conf:.2f}')
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
