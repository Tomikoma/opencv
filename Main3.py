import cv2
import numpy as np
import scipy.ndimage.measurements as msr
from imutils import rotate

def draw_rectangle(img,l): #https://www.programcreek.com/python/example/104526/scipy.ndimage.measurements.label -> draw_labeled_bboxes
    bbox=0
    for objects in range(0,len(msr.find_objects(l))):
        nonzero = (l == 1).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 2)
    return img

def to_gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


def bbox_f(img,l):
    bboxes=list()
    for objects in range(0,len(msr.find_objects(l))):
        nonzero = (l == 1).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)

    return bboxes

def crop_image(base,bbox):
    img=base[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
    return img

image=cv2.imread("tehen.png")
modified = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower = np.array([25, 75, 150], dtype="uint8")
upper = np.array([36, 255, 255], dtype="uint8")
mask = cv2.inRange(modified, lower, upper)
output = cv2.bitwise_and(modified, image, mask=mask)
BGRgrayImage=cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
RGBgrayImage=cv2.cvtColor(output,cv2.COLOR_RGB2GRAY)
c,l=cv2.connectedComponents(BGRgrayImage)
#cv2.imshow('',draw_rectangle(image,l))
bboxes=bbox_f(to_gray(output),l)
crop=crop_image(image,bboxes[0])
laplacian = cv2.Laplacian(crop,cv2.CV_64F)
edges=cv2.Canny(crop,50,200)
cv2.imshow('e',edges)
cv2.waitKey(0)


cap=cv2.VideoCapture("20190208_075319.mp4")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

while(cap.isOpened()):
        _, frame=cap.read()
        frame = rotate(frame, -90)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(frame,frame, mask= mask)

        gray=to_gray(res)
        c, l = cv2.connectedComponents(gray,connectivity=8)
        cv2.putText(frame,str(c),(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(200,200,255))
        modframe=draw_rectangle(frame,l)
        out.write(modframe)
        bboxes=bbox_f(frame,l)
        cv2.imshow('res', modframe)
        ind=0
        for bbox in bboxes:

            if (isinstance(bbox, tuple)):
                    crop_img = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
                    if(crop_img.size>3000):
                        print(crop_img.size)
                        cv2.imshow("cropped" + str(ind), crop_img)
            ind=ind+1

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
out.release()
cv2.destroyAllWindows()
