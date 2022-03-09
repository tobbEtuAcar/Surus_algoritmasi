import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel,kernel),0)
    canny = cv2.Canny(blur, 100, 110)
    return canny

def region_of_interest(image):
    y,x = image.shape
    image[0:210,:] = 0
    image[:,0:380] = 0
    Roi = image
    return Roi

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),5)
    return line_image
            
def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    sl = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2),1)
        slope,intercept = parameters[0],parameters[1] # neg-> left, pos-> right
        sl.append((slope))
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_avg = np.average(left_fit,axis=0)
    right_fit_avg = np.average(right_fit,axis=0)
    left_line = make_coordinates(image,left_fit_avg)
    right_line = make_coordinates(image,right_fit_avg)
    return np.array([left_line,right_line])
    
    
def make_coordinates(image,line_parameters):
    slope,intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(2/4))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

    
image = cv2.imread("/home/feanor/Desktop/line_detection/line_test3.png") # read the image
shape = image.shape
copy_image = np.copy(image)
start = time.time() # start run time
edge_image = canny(image) # apply edge detection
Roi = region_of_interest(edge_image) # we don't need to see all images for lane detection
copy_Roi = np.copy(Roi)
lines = cv2.HoughLinesP(Roi, 2, np.pi/180, 100,np.array([]),minLineLength=40,maxLineGap=10)
averaged_lines = average_slope_intercept(copy_image,lines)
line_image = display_lines(copy_image,averaged_lines) # averaged_lines or lines
result = cv2.addWeighted(copy_image, 0.8, line_image, 1, 1)
end = time.time()
total_time = end - start
print("Run time:{:1.5f} millisec".format(total_time))
cv2.imshow("Result",result)
cv2.imshow("Line",line_image)
cv2.waitKey(0)
cv2.destroyAllWindows()