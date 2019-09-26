import cv2
import math
import numpy as np
import pytesseract
from PIL import Image
from imutils import rotate


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


img = cv2.imread("images/enar2.png")
cv2.imshow("orig", img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 75, 150)

#img = rotate(img, get_rotation_angle(edges))
(h, w) = img.shape[:2]
center = (w / 2, h / 2)
scale = 1.0

M = cv2.getRotationMatrix2D(center, get_rotation_angle(edges),scale)
#img = cv2.warpAffine(img, M, (w, h))


h = img.shape[0]
img= img[int(h/4):, :]
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
kernel = np.ones((1, 1), np.uint8)
#img = cv2.dilate(img, kernel, iterations=1)
#img = cv2.erode(img, kernel, iterations=1)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)  # thresholding
(h, w) = th3.shape
greater = h if h > w else w
ratio = 150 / greater
print("H: " + str(h) + ", W: " + str(w))
if h < 150 and w < 150:
    th3 = cv2.resize(th3, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)  # interpolation for better results
cv2.imshow("before blur", th3)
#th3 = cv2.medianBlur(th3, 3)
th3 = cv2.bilateralFilter(th3, 9, 50, 150)

str_ = pytesseract.image_to_string(th3, config='-c tessedit_char_whitelist=0123456789 --oem 1 --psm 6 --tessdata-dir /home/tom/PycharmProjects/opencv/')

print("---")
print(str_)
cv2.waitKey(0)
cv2.destroyAllWindows()

