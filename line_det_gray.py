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


def canny(image):
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    canny = cv2.Canny(blur, 100, 110)
    return canny


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


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    sl = np.zeros(100)
    counter = 0
    counter_left = 0
    counter_right = 0
    left_line = [0, 0, 0, 0]
    right_line = [0, 0, 0, 0]
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters[0], parameters[1]  # neg-> left, pos-> right
        sl[counter] = slope
        if slope < 0:
            left_fit.append((slope, intercept))
            counter_left += 1
        if slope > 0:
            right_fit.append((slope, intercept))
            counter_right += 1
        counter += 1
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    if counter_left > 0:
        left_line = make_coordinates(image, left_fit_avg)
    if counter_right > 0:
        right_line = make_coordinates(image, right_fit_avg)
    return np.array([left_line, right_line]), sl


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (1 / 2))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    # if x1 < 0:
    #     x1 = 0
    # elif x1 > 640:
    #     x1 = 640
    return np.array([x1, y1, x2, y2])


def find_mid_line(lines):
    lx1, ly1, lx2, ly2 = lines[0]
    rx1, ry1, rx2, ry2 = lines[1]
    mx1 = int((lx1 + rx1) / 2)
    my1 = int((ly1 + ry1) / 2)
    mx2 = int((lx2 + rx2) / 2)
    my2 = int((ly2 + ry2) / 2)
    return np.array([mx1, my1, mx2, my2])


def find_mid_coordinates(lines):
    x1, y1, x2, y2 = lines
    slope = (y2 - y1) / (x2 - x1)
    coord = np.zeros((2, y1 - y2))
    coord[0][0], coord[1][0] = x2, y2
    counter = 1
    for i in range(y2 + 1, y1):
        x = x2 + ((i - y2) / slope)
        coord[0][counter], coord[1][counter] = x, i
    return coord


def get_coordinate(gray):
    afterMedian = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = 150
    bin = afterMedian > thresh
    sk = dip.EuclideanSkeleton(bin, endPixelCondition='three neighbors')
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
    endpoint_coords.sort(key=lambda x:x[0])
    return endpoint_coords


# cap = cv2.VideoCapture("/home/feanor/Desktop/line_detection/test_video.mp4")
cap = cv2.VideoCapture("/home/feanor/Desktop/line_detection/test_video2.mp4")
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
    edge_image = canny(gray)  # apply edge detection
    # copy_edge = np.copy(edge_image)

    Roi = region_of_interest(fit)
    kernel = 9
    blur = cv2.GaussianBlur(Roi, (kernel, kernel), 0)
    copy_Roi = np.copy(blur)
    endpoint_coords = get_coordinate(copy_Roi)
    for i in range(0,len(endpoint_coords)):
        cv2.circle(copy_fit, (endpoint_coords[i][0].astype(int), endpoint_coords[i][1].astype(int)),5,(0,0,255),cv2.FILLED)
    print(endpoint_coords)
    """"
    lines = cv2.HoughLinesP(Roi, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=10)
    averaged_lines, slxd = average_slope_intercept(copy_frame, lines)
    middle_line = find_mid_line(averaged_lines)  # detect edilen line'ların orta noktası
    line_frame = display_lines(copy_frame, averaged_lines, middle_line)  # averaged_lines or lines
    result = cv2.addWeighted(copy_frame, 0.8, line_frame, 1, 1)
    # end = time.time()
    # total_time = end - start
    # cv2.imshow("Result",result)
    # cv2.imshow("Original",copy_frame)
    # cv2.imshow("Edge detection",copy_edge)
    # cv2.imshow("Result",result)
    cv2.imshow("Line", copy_edge)
    cv2.imshow("Roi", result)
    """
    end = time.time()
    total_time = end - start
    #print(total_time)
    cv2.imshow("Result", copy_fit)
    #cv2.imshow("Roi", Roi)
    # # cv2.imshow("Video",gray)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
