import cv2
import numpy as np
import diplib as dip

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
        if np.sum(sub_img) == 255*2:
            found = 0
            for i in range(0,len(endpoint_coords)):
                if endpoint_coords[i][0] == c:
                    avg = (endpoint_coords[i][1] + r)/2
                    endpoint_coords[i] = (endpoint_coords[i][0],avg)
                    found = 1
                    break
            if found == 0:
                endpoint_coords.append((c,r))
    return endpoint_coords


cap = cv2.VideoCapture("park.mp4")
success, frame = cap.read()
frame = frame[24:504, :]

frame = frame[250:, :]
grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
afterMedian = cv2.bilateralFilter(grey, 9, 75, 75)
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
    if np.sum(sub_img) == 255*2:
        found = 0
        for i in range(0,len(endpoint_coords)):
            if endpoint_coords[i][0] == c:
                avg = (endpoint_coords[i][1] + r)/2
                endpoint_coords[i] = (endpoint_coords[i][0],avg)
                found = 1
                break
        if found == 0:
            endpoint_coords.append((c,r))

for i in range(0,len(endpoint_coords)):
    cv2.circle(frame, (endpoint_coords[i][0].astype(int), endpoint_coords[i][1].astype(int)),5,(0,0,255),cv2.FILLED)
print(endpoint_coords)
cv2.imshow("frame", frame)
cv2.waitKey(0)

