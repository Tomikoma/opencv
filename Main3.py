import cv2
import numpy as np
import scipy.ndimage.measurements as msr
import pytesseract as pt
from imutils import rotate
from imutils import resize




def image_to_string(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #th3 = cv2.Canny(img, 100, 200)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    th3 = cv2.resize(th3, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    th3 = cv2.medianBlur(th3, 3)
    # return (pt.image_to_string(th3, config='--psm 6 --oem 0 tessedit_char_whitelist=0123456789')), th3
    return (pt.image_to_string(th3, config='-c tessedit_char_whitelist=0123456789 --oem 0 --psm 6')), th3


def draw_rectangle(img,
                   l):  # https://www.programcreek.com/python/example/104526/scipy.ndimage.measurements.label -> draw_labeled_bboxes
    bbox = 0
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

        if (bbox[1][1] - bbox[0][1] > 30 and bbox[1][0] - bbox[0][0] > 30):
            bboxes.append(bbox)

    return bboxes


def crop_image(frame, bbox):
    img = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
    return img


image = cv2.imread("tehen.png")
# modified = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower = np.array([25, 75, 150], dtype="uint8")
upper = np.array([36, 255, 255], dtype="uint8")

# mask = cv2.inRange(modified, lower, upper)
# output = cv2.bitwise_and(modified, image, mask=mask)
# BGRgrayImage=cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
# RGBgrayImage=cv2.cvtColor(output,cv2.COLOR_RGB2GRAY)
# c,l=cv2.connectedComponents(BGRgrayImage)
# cv2.imshow('',draw_rectangle(image,l))
# bboxes=bbox_f(to_gray(output),l)
# crop=crop_image(image,bboxes[0])
# laplacian = cv2.Laplacian(crop,cv2.CV_64F)
# edges=cv2.Canny(crop,50,200)
# cv2.imshow('e',edges)
# cv2.waitKey(0)


# cap = cv2.VideoCapture("20190208_075319.mp4")
cap = cv2.VideoCapture("20190208_075319.mp4")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (frame_width, frame_height))
good = 0
bad = 0
word_dict = {}
lineCount = 0
lineArray = []
while cap.isOpened():
    _, frame = cap.read()
    frame = rotate(frame, -90)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    gray = to_gray(res)
    c, l = cv2.connectedComponents(gray, connectivity=8)
    cv2.putText(frame, str(c), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255))
    modframe = draw_rectangle(frame, l)
    # out.write(frame)
    bboxes = get_bbox_of_img(frame, l)
    cv2.imwrite('python.png', modframe)

    ind = 0

    for bbox in bboxes:

        if (isinstance(bbox, tuple)):

            # crop_img = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
            crop_img = crop_image(frame, bbox)
            if (crop_img.shape[0] > 30 and crop_img.shape[1] > 30):
                #print('Size: ' + str(crop_img.size) + ', H: ' + str(crop_img.shape[0]) + ', W: ' + str(crop_img.shape[0]))
                # print(crop_img.size)
                ocr_string, th3 = image_to_string(crop_img)
                if (ocr_string):
                    splitted_str = ocr_string.split("\n")
                    ocr_string = 'Recognized:\n' + ocr_string + '\nLength: ' + str(len(ocr_string)) + '  \n END'
                    print(ocr_string)
                    is_len_five = False
                    for _str in splitted_str:
                        if len(_str) == 5:
                            is_len_five = True
                    if (is_len_five):
                        print('Recognized:')
                        for _str in splitted_str:

                            if (len(_str) == 5):
                                if(len(_str.split(' ')) ==1):
                                    print(_str)
                                    if _str not in word_dict:
                                        word_dict[_str] = 1
                                    else:
                                        word_dict[_str] = word_dict.get(_str) + 1
                        print('END')

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
                edges = cv2.Canny(crop_img,100,150)
                edges = cv2.bitwise_not(edges)
                lines = cv2.HoughLines(edges,1,np.pi/180,150)


                if lines is not None:
                    lineCount = lineCount + 1
                    lineArray.append(lines)
                    for rho, theta in lines[0]:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))

                        cv2.line(edges, (x1, y1), (x2, y2), (150, 150, 255), 2)
                cv2.imshow('cro',edges)
                #cv2.imshow("cropped" + str(ind), th3)
        ind = ind + 1
    cv2.imshow('res', resize(modframe,1200,1200))
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        # out.release()
        # print('good: ' + str(good) + ', bad: ' + str(bad))
        print(word_dict)
        print(lineCount)
        print("lineArray:")
        print(lineArray)
        break
cv2.destroyAllWindows()
