import cv2
import numpy as np
import scipy.ndimage.measurements as msr
from imutils import rotate


def draw_labeled_bboxes(img, labels):
    """
        Draw the boxes around detected object.
    """
    # Iterate through all detected cars
    for car_number in range(1, len(msr.find_objects(labels[0])) ):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    return img


def to_gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


lower = np.array([25, 75, 150], dtype="uint8")
upper = np.array([36, 255, 255], dtype="uint8")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1920,1080))

cap=cv2.VideoCapture("20190208_075319.mp4")

index=0
sec=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frame = rotate(frame, -90)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        grayimg = to_gray(res)
        l, n = msr.label(grayimg)
        objects = msr.find_objects(l)


        # write the flipped frame
        s=draw_labeled_bboxes(frame,l)
        out.write(s)
        if(index<=30):
            index=index+1
        else:
            index=0
            sec=sec+1
            print(sec)

        cv2.imshow('frame',draw_labeled_bboxes(frame,[l]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
