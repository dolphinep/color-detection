import cv2
import numpy as np
import matplotlib.pyplot as plt

min_YCrCb = np.array([120,133,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)

def preprocess(action_frame):

    blur = cv2.GaussianBlur(action_frame, (3,3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

    lower_color = np.array([108, 23, 82])
    upper_color = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower_color, upper_color)
    blur = cv2.medianBlur(mask, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    hsv_d = cv2.dilate(blur, kernel)

    return hsv_d

# Get pointer to video frames from primary device
image = cv2.imread("IU_.jpg")
imageYCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
skinYCrCb = cv2.bitwise_and(image, image, mask = skinRegionYCrCb)
cv2.imshow("IU", np.hstack([image,skinYCrCb]))


#try preprocess
#cv2.imshow("IU", preprocess(image))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
