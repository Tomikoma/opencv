import cv2
import numpy as np
from imutils import rotate
from imutils import resize
import utils
import sys
import datetime
import subprocess


def main():
    lower = np.array([25, 75, 85], dtype="uint8")
    upper = np.array([30, 255, 255], dtype="uint8")

    if len(sys.argv) == 1:
        print("NO VIDEO FILE", flush=True)
        exit(0)
    video = sys.argv[1]
    rotating_angle = 0
    if len(sys.argv) > 2:
        rotating_angle = int(sys.argv[2])
    cap = cv2.VideoCapture(video)
    time = 0 * 1000
    cap.set(cv2.CAP_PROP_POS_MSEC, time)

    blank_image = np.zeros([200, 200, 3], dtype=np.uint8)
    blank_image.fill(255)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    filename = 'enarvideo' + datetime.datetime.now().strftime("%Y_%m%d_%H%M%S") + '.mp4'
    out = cv2.VideoWriter(filename, fourcc, float(60), (200, 200))

    dict_array = []
    frame_counter = 0
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    cap.get(cv2.CAP_PROP_POS_MSEC)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
    cap.set(cv2.CAP_PROP_POS_MSEC, time)
    while cap.isOpened():
        current_time = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        print(str(round(cap.get(cv2.CAP_PROP_POS_FRAMES)/cap.get(cv2.CAP_PROP_FRAME_COUNT), 3)), flush=True)
        # print(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time = int(current_time - (time / 1000))
        if len(dict_array) == current_time:
            dict_array.append(dict())
        ret, frame = cap.read()
        if not ret:
            break
        frame = rotate(frame, angle=rotating_angle)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        # cv2.imshow("+",resize(res, 1200, 800))

        gray = utils.to_gray(res)
        output = cv2.connectedComponentsWithStats(gray, 8, cv2.CV_32S)
        nlabels = output[0]
        stats = output[2]
        cv2.putText(frame, str(nlabels), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255))
        # l = msr.find_objects()
        modframe = utils.draw_rectangle(nlabels, frame, stats)
        # out.write(frame)
        bboxes = utils.get_bounding_rect(nlabels, stats)
        ind = 0
        frame_counter = frame_counter + 1
        for bbox in bboxes:

            if (isinstance(bbox, tuple)):
                crop_img = utils.crop_image(frame, bbox)

                if (crop_img.shape[0] > 30 and crop_img.shape[1] > 30):
                    # print('Size: ' + str(crop_img.size) + ', H: ' + str(crop_img.shape[0]) + ', W: ' + str(crop_img.shape[0]))
                    ocr_string, th3 = utils.image_to_string(crop_img)
                    if (ocr_string):
                        blank_image = cv2.resize(crop_img, (200, 200))
                        splitted_str = ocr_string.split("\n")
                        ocr_string = 'Recognized:\n' + ocr_string + '\nLength: ' + str(len(ocr_string)) + '  \n END'
                        #print(ocr_string, flush=True)

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
                    cv2.imshow('cro', th3)
                    # cv2.imshow("cropped" + str(ind), th3)
            ind = ind + 1

        out.write(blank_image)
        if current_time > 3:
            md = utils.merge_dicts(dict_array[current_time], dict_array[current_time - 1], dict_array[current_time - 2],
                                   dict_array[current_time - 3], dict_array[current_time - 4])
            enar4, enar5 = utils.get_enar_from_dict(md)

            if enar4 != "" and enar5 != "":
                cv2.putText(modframe, "Enar: " + enar4 + "(" + enar5[-1] + ")", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (200, 200, 255))

        cv2.imshow('res', resize(modframe, 1200, 1200))
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    out.release()
    dc = 0
    for d in dict_array:
        #print(dc, flush=True)
        dc = dc + 1
        #print(d, flush=True)

    # writing to file
    if len(dict_array) > 1:
        with open("enar" + datetime.datetime.now().strftime("%Y_%m%d_%H%M%S") + ".txt", "w") as f:
            i = 0
            for _ in dict_array:
                i = i + 1
                if i == 1:
                    m = utils.merge_dicts(dict_array[i], dict_array[i - 1])
                    #print(m)
                    enar4, enar5 = utils.get_enar_from_dict(m)
                    #print(enar4, enar5)
                    f.write(str(i) + " " + enar5 + "\n")
                elif i == len(dict_array):
                    m = utils.merge_dicts(dict_array[i - 1], dict_array[i - 2])
                    #print(m)
                    enar4, enar5 = utils.get_enar_from_dict(m)
                    #print(enar4, enar5)
                    f.write(str(i) + " " + enar5 + "\n")
                else:
                    m = utils.merge_dicts(dict_array[i], dict_array[i - 1], dict_array[i - 2])
                    #print(m)
                    enar4, enar5 = utils.get_enar_from_dict(m)
                    #print(enar4, enar5)
                    f.write(str(i) + " " + enar5 + "\n")

    command = "ffmpeg -i {} {}.mp4".format(filename,filename.split(".")[0] + "converted")
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print('FAIL:\ncmd:{}\noutput:{}'.format(e.cmd, e.output))


if __name__ == '__main__':
    main()
    print("DONE", flush=True)
