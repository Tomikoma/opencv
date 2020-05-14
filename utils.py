import pytesseract as pt
import math
import cv2
import numpy as np

# in progress
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
            #    y = y - 1
            elif edges.item(y-1,x-1) == 255:
                y = y - 1
                x = x - 1
            else:
                break

    angle = math.degrees(math.atan2(y_start-y,x_start-x))
    print("ANGLE: ", angle)
    return angle


def get_file_extension(path: str) -> str:
    return path.split(".")[-1]


# return recognized string
def image_to_string(img, lang, oem, psm) -> str:

    if len(img) == 0:
        return ""
    # converting image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # adaptive thresholding
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    # getting height and width
    (h, w) = thresh.shape
    greater = h if h > w else w
    ratio = 150/greater
    # if too small, interplation for better results
    if h < 150 and w < 150:
        thresh = cv2.resize(thresh, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)  # interpolation for better results
    # blurring for better result
    thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
    # setting config
    config = "-l {} --oem {} --psm {}".format(lang, oem, psm)
    # calling pytesseract image_to_string function
    # with the threshold of image and given config
    return pt.image_to_string(thresh, config=config)


# returns a list of bounding boxes (bboxes)
def get_bounding_boxes(nlabels, stats) -> list:
    bboxes = list()
    # iterating over the labels
    for label in range(1, nlabels):
        # getting stats of the label
        x = stats[label, cv2.CC_STAT_LEFT] - 3
        y = stats[label, cv2.CC_STAT_TOP] - 3
        width = stats[label, cv2.CC_STAT_WIDTH] + 3
        height = stats[label, cv2.CC_STAT_HEIGHT] + 3
        # constructing bounding box from stats
        bbox = (x, y), (x + width, y + height)
        # appending to list if not too small
        if width > 30 and height > 30:
            bboxes.append(bbox)

    return bboxes


# returns a cropped image
def crop_image(frame, bbox) -> np.ndarray:
    img = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
    return img


def merge_dicts(*dicts: dict) -> dict:
    # initialize empty dict
    merged_dict = dict()
    # iterating over the arguments
    for mydict in dicts:
        # typecheck
        if isinstance(mydict, dict):
            # iterating over one dict
            for key, value in mydict.items():
                # addition of number of recognitions
                if key in merged_dict:
                    merged_dict[key] = merged_dict[key] + value
                else:
                    merged_dict[key] = value
    return merged_dict


def get_enar_from_dict(enar_dict: dict) -> str:
    # typecheck
    if isinstance(enar_dict, dict):
        # initializing variables
        enar4 = ""
        enar4_num = enar5_num = -1
        # iterating over the given dict
        for key, value in enar_dict.items():
            if len(key) == 4:
                if value > enar4_num:
                    enar4 = key
                    enar4_num = value
        enar5 = enar4
        for key, value in enar_dict.items():
            if len(key) == 5 and key[:4] == enar4:
                if value > enar5_num:
                    enar5 = key
                    enar5_num = value
        # returning enars with the most occurrences
        return enar5
    else:
        return ""
