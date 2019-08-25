import scipy.ndimage.measurements as msr
import pytesseract as pt
import math
import cv2
import numpy as np

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

# converting image to string
def image_to_string(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # changing colorspace to gray (needed for processing)
    edges = cv2.Canny(img, 75, 150)
    cv2.imshow("edges", edges)
    #img = rotate(img, get_rotation_angle(edges))
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)  # thresholding
    th3 = cv2.resize(th3, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # interpolation for better results
    th3 = cv2.medianBlur(th3, 3)
    #th3 = cv2.bilateralFilter(th3, 9, 50, 150)
    return (pt.image_to_string(th3, config='-c tessedit_char_whitelist=0123456789 --oem 0 --psm 6')), th3


# function for drawing rectangle (visualization)
def draw_rectangle(img, l):  # https://www.programcreek.com/python/example/104526/scipy.ndimage.measurements.label -> draw_labeled_bboxes
    for objects in range(1, len(msr.find_objects(l))):
        nonzero = (l == 1).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bbox = (bbox[0][0] - 3, bbox[0][1] - 3), (bbox[1][0] + 3, bbox[1][1] + 3)
        if (bbox[1][1] - bbox[0][1] > 20 and bbox[1][0] - bbox[0][0] > 20):
            cv2.rectangle(img, (bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]), (0, 0, 255), 1)
    return img


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def get_bbox_of_img(img, l):
    bboxes = list()
    for objects in range(0, len(msr.find_objects(l))):
        nonzero = (l == 1).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bbox = (bbox[0][0] - 3, bbox[0][1] - 3), (bbox[1][0] + 3, bbox[1][1] + 3)

        if (bbox[1][1] - bbox[0][1] > 25 and bbox[1][0] - bbox[0][0] > 25):
            bboxes.append(bbox)

    return bboxes


def crop_image(frame, bbox):
    img = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
    return img


def merge_dicts(*argv):
    merged_dict = dict()
    for arg in argv:
        if isinstance(arg, dict):
            for key, value in arg.items():
                if key in merged_dict:
                    merged_dict[key] = merged_dict[key] + value
                else:
                    merged_dict[key] = value
    return merged_dict


def get_enar_from_dict(enar_dict):
    if isinstance(enar_dict, dict):
        enar4 = enar5 = ""
        enar4_num = enar5_num = -1
        for key, value in enar_dict.items():
            if len(key) == 4:
                if value > enar4_num:
                    enar4 = key
                    enar4_num = value

        for key, value in enar_dict.items():
            if len(key) == 5 and key[:4] == enar4:
                if value > enar5_num:
                    enar5 = key
                    enar5_num = value
        return enar4, enar5
    else:
        return "----", "-----"
