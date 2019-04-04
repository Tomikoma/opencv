import cv2
from imutils import rotate


cmyk_scale = 100

def rgb_to_cmyk(r,g,b):
    if (r == 0) and (g == 0) and (b == 0):
        # black
        return 0, 0, 0, cmyk_scale

    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / 255.
    m = 1 - g / 255.
    y = 1 - b / 255.

    # extract out k [0,1]
    min_cmy = min(c, m, y)
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy

    # rescale to the range [0,cmyk_scale]
    return c*cmyk_scale, m*cmyk_scale, y*cmyk_scale, k*cmyk_scale

cap = cv2.VideoCapture('20190208_075319.mp4')

while cap.isOpened():
     ret, frame = cap.read()

     if ret:
         frame = rotate(frame, -90)
         imageWidth = frame.shape[1]  # Get image width
         imageHeight = frame.shape[0]  # Get image height

         xPos, yPos = 0, 0

         while xPos < imageWidth:  # Loop through rows
             while yPos < imageHeight:  # Loop through collumns
                 k = rgb_to_cmyk(frame.item(yPos, xPos, 2), frame.item(yPos, xPos, 1), frame.item(yPos, xPos, 0))

                 if (k[2] < 50):
                     frame.itemset((yPos, xPos, 0), 0)  # Set B to 255
                     frame.itemset((yPos, xPos, 1), 0)  # Set G to 255
                     frame.itemset((yPos, xPos, 2), 0)  # Set R to 255

                 yPos = yPos + 1  # Increment Y position by 1

             yPos = 0
             xPos = xPos + 1  # Increment X position by 1
         cv2.imshow('frame', frame)
         if cv2.waitKey(1) & 0xFF == ord('q'):
             break

     else:
         break

cap.release()
cv2.destroyAllWindows()

