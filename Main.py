import cv2
import numpy as np
from imutils import rotate
from imutils import resize
import utils
import sys
import datetime
import subprocess
import platform

# initializing lower boundary of color yellow
lower = np.array([25, 75, 85], dtype="uint8")
# initializing upper boundary of color yellow
upper = np.array([30, 255, 255], dtype="uint8")
# needed for filtering


SUPPORTED_EXTENSIONS = ["mp4", "mpeg"]


def main(argv: list):
    # argument check
    if len(argv) == 1:
        print("NO VIDEO FILE", flush=True)
        exit(0)
    # set videopath
    video = argv[1]
    # get file extension
    extension = utils.get_file_extension(video).lower()
    # extension check
    if extension not in SUPPORTED_EXTENSIONS:
        print("NOT SUPPORTED EXTENSION")
        exit(0)
    rotating_angle = 0
    # set rotating angle if second parameter given
    if len(argv) > 2:
        rotating_angle = int(argv[2])
    # construct VideoCapture object
    cap = cv2.VideoCapture(video)
    # initializing base white image
    image_to_save = np.zeros([200, 200, 3], dtype=np.uint8)
    image_to_save.fill(255)
    # 4-character code of codec for writing video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    datetime_stamp = datetime.datetime.now()\
        .strftime("%Y_%m%d_%H%M%S")
    # temporary video file
    temp_video = "temp_video.mp4"
    filename = 'enarvideo' + datetime_stamp + '.mp4'
    fps = cap.get(cv2.CAP_PROP_FPS)
    # construct VideoWriter object
    # needed to make video from cropped enarnumbers
    out = cv2.VideoWriter(temp_video, fourcc, float(fps), (200, 200))
    # initialize array for enardicts
    dict_array = []
    # processing the video
    while True:
        # get one frame from VideoCapture object
        ret, frame = cap.read()
        # if no more frame, stop processing
        if not ret:
            break

        # getting current_time in sec
        current_time = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        # this print needed for the wrapper app
        print(str(round(cap.get(cv2.CAP_PROP_POS_FRAMES)/cap.get(cv2.CAP_PROP_FRAME_COUNT), 3)), flush=True)
        # append new dict
        if len(dict_array) == current_time:
            dict_array.append(dict())
        # rotating frame
        frame = rotate(frame, angle=rotating_angle)
        # converting image to HSV colorspace
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # creating mask for filtering
        mask = cv2.inRange(hsv, lower, upper)
        # filtering frame with mask
        res = cv2.bitwise_and(frame, frame, mask=mask)

        # converting to grayscale
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        # searching connected components
        output = cv2.connectedComponentsWithStats(gray, 8)
        # number of labels
        nlabels = output[0]
        # stats of the components (width,height etc.)
        stats = output[2]
        modframe = frame.copy()
        # getting the bounding boxes
        bboxes = utils.get_bounding_boxes(nlabels, stats)
        # iterating over the bounding boxes
        for bbox in bboxes:
            # drawing rectangle to displayed frame
            cv2.rectangle(modframe, (bbox[0][0], bbox[0][1]),
                          (bbox[1][0], bbox[1][1]), (0, 0, 255), 1)
            # cropping image
            crop_img = utils.crop_image(frame, bbox)
            # getting string from image
            ocr_string = utils.image_to_string(crop_img, "digits", 1, 6)
            # check
            if (ocr_string):
                # saving cropped image to write out later
                image_to_save = cv2.resize(crop_img, (200, 200))
                # splitting str
                splitted_str = ocr_string.split("\n")
                # ocr_string = 'Recognized:\n' + ocr_string + '\nLength: ' + str(len(ocr_string)) + '  \n END'

                # iterating over the lines
                for _str in splitted_str:
                    # if only one word in the line and longer then 3
                    if len(_str.split(' ')) == 1 and len(_str) > 3:
                        # adding to dict
                        if _str[:4] not in dict_array[current_time]:
                            dict_array[current_time][_str[:4]] = 1
                        else:
                            dict_array[current_time][_str[:4]] = dict_array[current_time].get(_str[:4]) + 1
                    # if only one word in the line and longer then 4
                    if len(_str.split(' ')) == 1 and len(_str) > 4:
                        # adding to dict
                        if _str[:5] not in dict_array[current_time]:
                            dict_array[current_time][_str[:5]] = 1
                        else:
                            dict_array[current_time][_str[:5]] = dict_array[current_time].get(_str[:5]) + 1
        out.write(image_to_save)
        # if current_time > 3:
        #     md = utils.merge_dicts(dict_array[current_time], dict_array[current_time - 1], dict_array[current_time - 2],
        #                            dict_array[current_time - 3], dict_array[current_time - 4])
        #     enar5 = utils.get_enar_from_dict(md)
        #
        #     if enar5 != "":
        #         cv2.putText(modframe, "Enar: " + enar5, (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #                     (200, 200, 255))

        cv2.imshow('video', resize(modframe, 1200, 1200))
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    cap.release()
    out.release()

    # writing to file

    # checking size
    if len(dict_array) > 1:
        # opening file to write
        with open("enar" + datetime_stamp + ".txt", "w") as f:
            i = 0
            # iterating over the array
            for _ in dict_array:
                i = i + 1
                # merging dicts
                # and getting enarnumber
                if i == 1:
                    m = utils.merge_dicts(dict_array[i],
                                          dict_array[i - 1])
                    #print(m)
                    enar5 = utils.get_enar_from_dict(m)
                    #print(enar4, enar5)
                elif i == len(dict_array):
                    m = utils.merge_dicts(dict_array[i - 1],
                                          dict_array[i - 2])
                    #print(m)
                    enar5 = utils.get_enar_from_dict(m)
                    #print(enar4, enar5)
                else:
                    m = utils.merge_dicts(dict_array[i], dict_array[i - 1],
                                          dict_array[i - 2])
                    #print(m)
                    enar5 = utils.get_enar_from_dict(m)
                    #print(enar4, enar5)
                # writing time and recognized enar to file
                f.write(str(i) + " " + (enar5 or "") + "\n")
        # converting video with ffmpeg
        command = "ffmpeg -i {} {}\n".format(
            temp_video, filename)
        try:
            # running command
            output = subprocess.check_output(
                command, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            print('FAIL:\ncmd:{}\noutput:{}'.format(e.cmd, e.output))
    # and deleting temporary video file
    delete_command = "del" if platform.system() == "Windows" else "rm"
    subprocess.call("{} {}".format(delete_command, temp_video), shell=True)


# calling main
if __name__ == '__main__':
    main(sys.argv)
    print("DONE", flush=True)
