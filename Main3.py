import cv2
import numpy as np
import scipy.ndimage.measurements as msr
import pytesseract as pt
import math
from imutils import rotate
from imutils import resize

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
    img = rotate(img, get_rotation_angle(edges))
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)  # thresholding
    th3 = cv2.resize(th3, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # interpolation for better results
    th3 = cv2.medianBlur(th3, 3)
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


image = cv2.imread("ot.png")

# lower = np.array([25, 75, 85], dtype="uint8")
# upper = np.array([36, 255, 255], dtype="uint8")

lower = np.array([25, 75, 85], dtype="uint8")
upper = np.array([30, 255, 255], dtype="uint8")

cap = cv2.VideoCapture("20190208_075319.mp4")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (frame_width, frame_height))
good = 0
bad = 0
word_dict = {}
lineCount = 0
lineArray = []
dict_counter = 0
frame_counter = 0
counter_max_value = 30
dict_array = []
while cap.isOpened():
    if frame_counter == 0:
        dict_array.append(dict())
    frame_counter = frame_counter + 1
    print("FRAME COUNTER",frame_counter)
    _, frame = cap.read()
    frame = rotate(frame, -90)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imshow("+",resize(res, 1200, 800))

    gray = to_gray(res)
    c, l = cv2.connectedComponents(gray, connectivity=8)


    cv2.putText(frame, str(c), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255))
    #l = msr.find_objects()
    modframe = draw_rectangle(frame, l)
    # out.write(frame)
    bboxes = get_bbox_of_img(frame, l)
    cv2.imwrite('python.png', modframe)

    ind = 0

    for bbox in bboxes:

        if (isinstance(bbox, tuple)):

            # crop_img = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
            crop_img = crop_image(frame, bbox)
            if (crop_img.shape[0] > 25 and crop_img.shape[1] > 25):
                #print('Size: ' + str(crop_img.size) + ', H: ' + str(crop_img.shape[0]) + ', W: ' + str(crop_img.shape[0]))
                # print(crop_img.size)
                ocr_string, th3 = image_to_string(crop_img)
                if (ocr_string):
                    splitted_str = ocr_string.split("\n")
                    ocr_string = 'Recognized:\n' + ocr_string + '\nLength: ' + str(len(ocr_string)) + '  \n END'
                    print(ocr_string)

                    for _str in splitted_str:
                        if len(_str.split(' ')) == 1:
                            if len(_str) > 3:
                                if _str[:4] not in word_dict:
                                    word_dict[_str[:4]] = 1
                                else:
                                    word_dict[_str[:4]] = word_dict.get(_str[:4]) + 1
                            if len(_str) > 4:
                                if _str[:5] not in word_dict:
                                    word_dict[_str[:5]] = 1
                                else:
                                    word_dict[_str[:5]] = word_dict.get(_str[:5]) + 1

                    for _str in splitted_str:
                        if len(_str.split(' ')) == 1:
                            if len(_str) > 3:
                                if _str[:4] not in dict_array[dict_counter]:
                                    dict_array[dict_counter][_str[:4]] = 1
                                else:
                                    dict_array[dict_counter][_str[:4]] = dict_array[dict_counter].get(_str[:4]) + 1
                            if len(_str) > 4:
                                if _str[:5] not in dict_array[dict_counter]:
                                    dict_array[dict_counter][_str[:5]] = 1
                                else:
                                    dict_array[dict_counter][_str[:5]] = dict_array[dict_counter].get(_str[:5]) + 1
                else:
                    ocr_string = 'Nothing can be recognized'
                # print('Recognized: ', ocr_string   )
                if(len(word_dict) == 0 ):
                    cv2.putText(modframe, 'Nothing can be recognized', (40, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255))
                else:
                    recognized_word = ''
                    word_count = -1
                    for word in word_dict.keys():
                        if word_dict[word] >= word_count:
                            recognized_word = word
                            word_count = word_dict[word]

                    cv2.putText(modframe, 'Recognized word: ' + recognized_word, (40, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255))
                #cv2.putText(modframe, ocr_string, (40, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255))
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                crop_img = cv2.resize(crop_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                cv2.imshow('cro',th3)
                # cv2.imshow("cropped" + str(ind), th3)
        ind = ind + 1

    if dict_counter > 3:
        md = merge_dicts(dict_array[dict_counter],dict_array[dict_counter-1],dict_array[dict_counter-2],dict_array[dict_counter-3],dict_array[dict_counter-4])
        enar4 = enar5 = ""
        enar4_num = enar5_num = -1
        for key, value in md.items():
            if len(key) == 4:
                if value > enar4_num:
                    enar4 = key
                    enar4_num = value

        for key, value in md.items():
            if len(key) == 5 and key[:4] == enar4:
                if value > enar5_num:
                    enar5 = key
                    enar5_num = value

        if enar4 != "" and enar5 != "" and enar4_num != -1 and enar5_num != -1:
            cv2.putText(modframe, "Enar: " + enar4 + "(" + enar5[-1] + ")", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (200, 200, 255))

    if frame_counter == counter_max_value:
        frame_counter = 0
        dict_counter = dict_counter + 1
    cv2.imshow('res', resize(modframe,1200,1200))
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        # out.release()
        # print('good: ' + str(good) + ', bad: ' + str(bad))
        print(word_dict)
        print(lineCount)
        # print("lineArray:")
        # print(lineArray)
        break
cv2.destroyAllWindows()
enar4 = enar5 = ""
enar4_num = enar5_num = -1
for key, value in word_dict.items():
    if len(key) == 4:
        if value > enar4_num:
            enar4 = key
            enar4_num = value

for key, value in word_dict.items():
    if len(key) == 5 and key[:4] == enar4:
        if value > enar5_num:
            enar5 = key
            enar5_num = value

print("Counter value: ",dict_counter)

dc=0
for d in dict_array:
    print(dc)
    dc = dc + 1
    print(d)
print("Enar: " + enar4 + "(" + enar5[-1] + ")")
