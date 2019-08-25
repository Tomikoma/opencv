import cv2
import numpy as np
from imutils import rotate

if __name__ == '__main__':
    image=cv2.imread("tehen.png")
    #base=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    modified = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    modified2=cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    #cv2.imshow("image", image)
    #cv2.imshow("base", base)
    #cv2.imshow("YCrCb", modified)
    #cv2.imshow("HSV", modified2)
    #cv2.waitKey(0)


    lower = np.array([25, 75, 150], dtype="uint8")
    upper = np.array([36, 255, 255], dtype="uint8")
    mask = cv2.inRange(modified, lower, upper)
    output = cv2.bitwise_and(modified, image, mask=mask)

    #start matlab engine
    eng = matlab.engine.start_matlab()
    img = matlab.double(output.tolist())
    grayImage=eng.rgb2gray(img)
    binaryImage = eng.im2bw(grayImage)
    label=eng.bwlabel(binaryImage)
    s=eng.regionprops(label)
    croppedImage=eng.imcrop(label, s['BoundingBox'])
    cv2.imshow("lol",croppedImage)
    print(eng.bwconncomp(label))
    #eng.vislabels(label,nargout=0)



    cap=cv2.VideoCapture("20190208_075319.mp4")
    index=0
    while(cap.isOpened()):
        _, frame=cap.read()
        frame = rotate(frame, -90)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(frame,frame, mask= mask)

        #cv2.imshow('frame', frame)
        #cv2.imshow('mask', mask)
        cv2.imshow('res', res)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

    #stop matlab engine
    eng.quit()
    #cv2.imshow("images", np.hstack([image,modified,modified2, output]))
    #cv2.waitKey(0)




