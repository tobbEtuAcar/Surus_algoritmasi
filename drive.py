#!/usr/bin/env python
#-*-coding: utf-8 -*-
 
import cv2
import numpy as np
import rospy
import time
import torch
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

model = torch.hub.load('/home/talhaunal/yolov5', 'custom', path='/home/talhaunal/yolov5/best_3Kasım.pt', source='local')
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# global speed
# global steering_angle	
steering_angle = 0.0
speed = 2

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
    movement(regions)
    
def movement(regions):

    global flag
    global counter
    global speed
    global steering_angle
    global MOD
    global MOD_PREV

    if MOD == 'SAG REFERANS':

        if regions['right'] > 0.3:
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

    elif MOD == 'SOL REFERANS':

        if regions['left'] > 0.3:
            speed = 2
            
            if regions['left'] > 0.99000 and regions['left'] < 1.05:   
                #print('DÜZ GİDECEN')
                steering_angle = 0.0
                speed = 2 
                flag = 0
                counter = 0

            elif regions['left'] > 2.1:  
                #print('HARD LEFT')
                steering_angle = 0.4
                speed = 1.8
                flag = 1

            elif regions['left'] > 1.05 and regions['left'] < 2.1 and flag == 1:  
                #print('HARD LEFT')
                steering_angle = 0.4
                speed = 1.8

            elif  regions['left'] > 1.05 and flag == 0: 
                #print('SOFT LEFT')  
                steering_angle = 0.02
                speed = 1.8   

            elif regions['left'] < 0.99000 and regions['right'] > 1.2:  
                #print('HARD RIGHT')
                steering_angle = -0.5
                speed = 1.8


            elif regions['left'] < 0.99000 and regions['left'] > 0.9  and flag == 0:  
                #print('SOFT RIGHT')
                steering_angle = -0.02
                speed = 1.8
    
        else: 
            # print('STOP!')
            speed = 0.0
            steering_angle = 0.0
        
    elif MOD == 'DURAK':
        speed = 0.0
        time.sleep(10)
        if(MOD_PREV != MOD):
            MOD = MOD_PREV
            MOD_PREV = 'SAG REFERANS'

    obj.drive.speed = speed
    obj.drive.steering_angle = steering_angle
    pub.publish(obj)

def cameraCallback(mesaj):
        bridge = CvBridge()
        foto = bridge.imgmsg_to_cv2(mesaj,"bgr8")
        results = model(foto)

        global MOD
        global MOD_PREV
        global DURAKModuBitisZamani

        print('MOD:', MOD)
        print('****************')
        
        for *box, conf, cls in results.pred[0]:

            if(results.names[int(cls)].find('stop1') != -1 and float(f'{conf:.2f}') > 0.83 and MOD != 'DURAK'):
                MOD_PREV = MOD
                MOD = 'DURAK'

                break
                # print(f'{conf:.2f}')
            elif(results.names[int(cls)].find('turnRight') != -1 and float(f'{conf:.2f}') > 0.6 and MOD != 'SAG REFERANS'):
                MOD_PREV = MOD
                MOD = 'SAG REFERANS'
                break

            elif(results.names[int(cls)].find('turnLeft') != -1 and float(f'{conf:.2f}') > 0.4 and MOD != 'SOL REFERANS'):
                MOD_PREV = MOD
                MOD = 'SOL REFERANS'
                break

            label = f'{results.names[int(cls)]} {conf:.2f}'
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

        cv2.imshow("Arac Kamerasi", results.render()[0])
        #cv2.imshow("Arac Kamerasi", foto)
        cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('drive', anonymous = True)
    rospy.Subscriber('/scan', LaserScan, lidar_callback)
    rospy.Subscriber("/camera/zed/rgb/image_rect_color", Image, cameraCallback)

    rospy.loginfo("Press CTRL + C for stopping the simulation")

    pub = rospy.Publisher("/vesc/high_level/ackermann_cmd_mux/output", AckermannDriveStamped, queue_size = 10)

    obj = AckermannDriveStamped()

    rospy.spin()
