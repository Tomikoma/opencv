import cv2
import math
import numpy as np
import pytesseract
from PIL import Image
from imutils import rotate

"""
cap = cv2.VideoCapture(0)

if cap.isOpened():
    ret, frame = cap.read()
else:
    ret = False

while ret:
    ret, frame = cap.read()

    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grey,50,250,apertureSize=5, L2gradient=True)
í
    lines = cv2.HoughLines(edges,1, np.pi/180,200)

    if lines is not None:
        for rho, theta, in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            pts1 = (int(x0 + 1000*(-b)),  int(y0 + 1000*(a)))
            pts2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(frame,pts1,pts2,(0,255,0), 1)

    cv2.imshow("Test", frame)

    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
cap.release()
"""

def get_rotation_angle(edges):
    left_is_smaller = True
    y = edges.shape[0]-1
    x = int(edges.shape[1]/2)
    while edges.item(y, x) != 255:
        #edges.itemset((y,x),255)
        y = y - 1
    y_start = y
    x_start = x

    y_left = edges.shape[0] - 1
    x_left = int(edges.shape[1] / 4)
    while edges.item(y_left, x_left) != 255:
        y_left = y_left - 1

    if y_left < y:
        left_is_smaller = False # problem with variable name

    if left_is_smaller:
        while edges.item(y,x) != 0:
            #print(y,x)
            if edges.item(y,x-1) == 255:
                x = x - 1
            #elif edges.item(y+1,x) == 255:
            #    y = y + 1
            elif edges.item(y+1,x-1) == 255:
                y = y + 1
                x = x - 1
            else:
                break
    else:
        while edges.item(y,x) != 0:
            #print(y,x)
            if edges.item(y,x-1) == 255:
                x = x - 1
            #elif edges.item(y-1,x) == 255:
             #   y = y - 1
            elif edges.item(y-1,x-1) == 255:
                y = y - 1
                x = x - 1
            else:
                break

    angle = math.degrees(math.atan2(y_start-y,x_start-x))
    print(angle)
    return angle



img = cv2.imread("enar.png")
cv2.imshow("orig", img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 75, 150)
#img = rotate(img, get_rotation_angle(edges))
h = img.shape[0]
img= img[int(h/3):, :]
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)  # thresholding
th3 = cv2.resize(th3, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # interpolation for better results
th3 = cv2.medianBlur(th3, 5)
str= pytesseract.image_to_string(th3, config='-c tessedit_char_whitelist=0123456789 --oem 0 --psm 6')
#ret, thresh = cv2.threshold(grey,127,255,cv2.THRESH_BINARY)

edges = cv2.Canny(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),75,150)
cv2.imshow("th3", th3)
cv2.imshow("enar", img)

print(str)
cv2.waitKey(0)
cv2.destroyAllWindows()

