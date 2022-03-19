import cv2
import numpy as np
import time
import diplib as dip
from functools import reduce
import operator
import math


def fit_image(image):
    image = image[24:504, :]
    return image


def region_of_interest(image):
    y, x = image.shape
    image[0:250, :] = 0
    image[290:480, :] = 0
    Roi = image
    return Roi


def display_lines(image, lines, mid_line):
    line_image = np.zeros_like(image)
    # display averaged lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    # display middle line
    x1, y1, x2, y2 = mid_line.reshape(4)
    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    # display car'line
    x1, y1, x2, y2 = 320, y1, 320, y2
    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return line_image


def get_coordinate(gray):
    afterMedian = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = 130
    bin = afterMedian > thresh
    sk = dip.EuclideanSkeleton(bin, endPixelCondition='one neighbor')
    sk = np.array(sk)
    sk = np.array(sk, dtype=np.uint8)
    sk *= 255
    (rows, cols) = np.nonzero(sk)

    # Initialize empty list of coordinates
    endpoint_coords = []

    # Loop through all non-zero pixels
    for (r, c) in zip(rows, cols):
        top = max(0, r - 1)
        right = min(sk.shape[1] - 1, c + 1)
        bottom = min(sk.shape[0] - 1, r + 1)
        left = max(0, c - 1)

        sub_img = sk[top: bottom + 1, left: right + 1]
        if np.sum(sub_img) == 255 * 2:
            found = 0
            for i in range(0, len(endpoint_coords)):
                if endpoint_coords[i][0] == c:
                    avg = (endpoint_coords[i][1] + r) / 2
                    endpoint_coords[i] = (endpoint_coords[i][0], avg)
                    found = 1
                    break
            if found == 0:
                endpoint_coords.append((c, r))
    endpoint_coords.sort(key=lambda x: x[0])
    return endpoint_coords


def left_right_coordinates(coords):
    length = len(coords)
    left = []
    right = []
    switch = 0
    for i in range(length - 1):
        if coords[i + 1][0] - coords[i][0] < 250:
            if switch == 0:
                left.append(coords[i])
            else:
                right.append(coords[i])
        else:
            if switch == 0:
                left.append(coords[i])
            else:
                right.append(coords[i])
            switch = 1
    if switch == 1:  # Son elemanı ekle.
        right.append(coords[length - 1])
    return left, right


def find_mid_coordinates(left_coords, right_coords):
    len_l, len_r = len(left_coords), len(right_coords)
    left_last, left_init = left_coords[len_l - 1], left_coords[0]
    right_last, right_init = right_coords[0], right_coords[len_r - 1]
    mid_coords = []
    mid_coords.append((np.array(left_init) + np.array(right_init)) // 2)
    mid_coords.append((np.array(left_last) + np.array(right_last)) // 2)
    return mid_coords


# cap = cv2.VideoCapture("/home/feanor/Desktop/line_detection/test_video.mp4")
cap = cv2.VideoCapture("/home/feanor/Desktop/line_detection/test_video2.mp4")
# cap = cv2.VideoCapture("park.mp4")
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
counter = 0

while True:
    _, frame = cap.read()
    counter += 1
    if counter == (frame_count - 1):
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fit = fit_image(gray)
    copy_fit = np.copy(fit)
    resolution = fit.shape
    copy_frame = np.copy(frame)
    start = time.time()  # start run time
    Roi = region_of_interest(fit)
    kernel = 9
    blur = cv2.GaussianBlur(Roi, (kernel, kernel), 0)
    copy_Roi = np.copy(blur)
    endpoint_coords = get_coordinate(copy_Roi)
    for i in range(0, len(endpoint_coords)):
        cv2.circle(copy_fit, (endpoint_coords[i][0].astype(int), endpoint_coords[i][1].astype(int)), 5, (0, 0, 255),
                   cv2.FILLED)
    #print(endpoint_coords)
    if len(endpoint_coords) == 0:
        print("DON")
    else:
        print("DUZ")
    left_coords, right_coords = left_right_coordinates(endpoint_coords)
    if len(left_coords) and len(right_coords) >= 2:
        mid_coords = find_mid_coordinates(left_coords, right_coords)
        for i in range(len(mid_coords)):
            cv2.circle(copy_fit, (mid_coords[i][0].astype(int), mid_coords[i][1].astype(int)), 5, (0, 0, 255),
                       cv2.FILLED)
    end = time.time()
    total_time = end - start
    # print(total_time)
    cv2.imshow("Result", copy_fit)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
