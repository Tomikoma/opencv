import cv2 as cv
import numpy as np
import pytesseract
from PIL import Image

img =cv.imread('ocr1_1.jpg')

img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#img = cv.medianBlur(img,5)

# global thresholding
ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)

ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

print(pytesseract.image_to_string(th3,config='--psm 6 --oem 1'))
cv.imshow('img',th3)
cv.waitKey(0)
