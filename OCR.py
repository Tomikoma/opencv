import cv2
import numpy as np
import pytesseract
from PIL import Image
from imutils import rotate

img =cv2.imread('ocr1forgatva.jpg')
img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
#                           cv2.THRESH_BINARY, 11, 2)
# print(pytesseract.image_to_string(th1, config='-c tessedit_char_whitelist=0123456789 --oem 0 --psm 6'))
# th1 = th1[int(th1.shape[0]/2):th1.shape[0],0:th1.shape[0]] for cropping images
# print(pytesseract.image_to_string(th1, config='-c tessedit_char_whitelist=0123456789 --oem 0 --psm 6'))
edges = cv2.Canny(th1,50,100)
lines = cv2.HoughLines(edges,1,np.pi/180,150)
for r, theta in lines[0]:
    # Stores the value of cos(theta) in a
    a = np.cos(theta)

    # Stores the value of sin(theta) in b
    b = np.sin(theta)

    # x0 stores the value rcos(theta)
    x0 = a * r

    # y0 stores the value rsin(theta)
    y0 = b * r

    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
    x1 = int(x0 + 1000 * (-b))

    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
    y1 = int(y0 + 1000 * (a))

    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
    x2 = int(x0 - 1000 * (-b))

    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
    y2 = int(y0 - 1000 * (a))

    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
    # (0,0,255) denotes the colour of the line to be
    # drawn. In this case, it is red.
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# All the changes made in the input image are finally
# written on a new image houghlines.jpg

cv2.imshow('img', cv2.resize(img,(700,700)))
cv2.waitKey(0)
