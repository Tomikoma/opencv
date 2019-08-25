import cv2
import numpy as np
import scipy.ndimage.measurements as msr
import pytesseract as pt
import math
from imutils import rotate
from imutils import resize
import utils


def main():
    lower = np.array([25, 75, 85], dtype="uint8")
    upper = np.array([30, 255, 255], dtype="uint8")

    video = "20190208_075319.mp4"
    cap = cv2.VideoCapture(video)

    # frame_width = int(cap.get(3))
    # frame_height = int(cap.get(4))
    # out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (frame_width, frame_height))

    dict_array = []
    while cap.isOpened():
        current_time = int(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
        if len(dict_array) == current_time:
            dict_array.append(dict())
        print("curr_time: " + str(current_time))
        _, frame = cap.read()

        if video == "20190208_075319.mp4":
            frame = rotate(frame, -90)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        # cv2.imshow("+",resize(res, 1200, 800))

        gray = utils.to_gray(res)
        components, labels = cv2.connectedComponents(gray, connectivity=8)


        cv2.putText(frame, str(components), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255))
        #l = msr.find_objects()
        modframe = utils.draw_rectangle(frame, labels)
        # out.write(frame)
        bboxes = utils.get_bbox_of_img(frame, labels)

        ind = 0

        for bbox in bboxes:

            if (isinstance(bbox, tuple)):
                crop_img = utils.crop_image(frame, bbox)
                if (crop_img.shape[0] > 25 and crop_img.shape[1] > 25):
                    #print('Size: ' + str(crop_img.size) + ', H: ' + str(crop_img.shape[0]) + ', W: ' + str(crop_img.shape[0]))
                    ocr_string, th3 = utils.image_to_string(crop_img)
                    if (ocr_string):
                        splitted_str = ocr_string.split("\n")
                        ocr_string = 'Recognized:\n' + ocr_string + '\nLength: ' + str(len(ocr_string)) + '  \n END'
                        print(ocr_string)

                        for _str in splitted_str:
                            if len(_str.split(' ')) == 1:
                                if len(_str) > 3:
                                    if _str[:4] not in dict_array[current_time]:
                                        dict_array[current_time][_str[:4]] = 1
                                    else:
                                        dict_array[current_time][_str[:4]] = dict_array[current_time].get(_str[:4]) + 1


                                if len(_str) > 4:
                                    if _str[:5] not in dict_array[current_time]:
                                        dict_array[current_time][_str[:5]] = 1
                                    else:
                                        dict_array[current_time][_str[:5]] = dict_array[current_time].get(_str[:5]) + 1
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    crop_img = cv2.resize(crop_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    cv2.imshow('cro',th3)
                    # cv2.imshow("cropped" + str(ind), th3)
            ind = ind + 1

        if current_time > 3:
            md = utils.merge_dicts(dict_array[current_time],dict_array[current_time-1],dict_array[current_time-2],dict_array[current_time-3],dict_array[current_time-4])
            enar4, enar5 = utils.get_enar_from_dict(md)

            if enar4 != "" and enar5 != "":
                cv2.putText(modframe, "Enar: " + enar4 + "(" + enar5[-1] + ")", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (200, 200, 255))

        cv2.imshow('res', resize(modframe,1200,1200))
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

    dc=0
    for d in dict_array:
        print(dc)
        dc = dc + 1
        print(d)




    # writing to file

    if len(dict_array) > 1:
        f = open("proba.txt", "w")
        i = 0
        for d in dict_array:
            i = i + 1
            enar4 = enar5 = ""
            if i == 1:
                m = utils.merge_dicts(dict_array[i], dict_array[i - 1])
                print(m)
                enar4, enar5 = utils.get_enar_from_dict(m)
                print(enar4,enar5)
                f.write(str(i) + " " + enar5 + "\n")
            elif i == len(dict_array):
                m = utils.merge_dicts(dict_array[i-1],dict_array[i-2])
                print(m)
                enar4, enar5 = utils.get_enar_from_dict(m)
                print(enar4, enar5)
                f.write(str(i) + " " + enar5 + "\n")
            else:
                m = utils.merge_dicts(dict_array[i],dict_array[i-1],dict_array[i-2])
                print(m)
                enar4, enar5 = utils.get_enar_from_dict(m)
                print(enar4, enar5)
                f.write(str(i) + " " + enar5 + "\n")
        f.close()


if __name__ == '__main__':
    main()
