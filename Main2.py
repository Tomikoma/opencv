import cv2
from imutils import rotate

cap = cv2.VideoCapture('20190208_075319.mp4')

while cap.isOpened():
     ret, frame = cap.read()

     if ret:
         frame = rotate(frame, -90)
         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
         y, cr, cb = cv2.split(gray) #csatornák szétválasztása
         print(y)
         break
         y[y > 200] = 255
         y[y < 200] = 0
         cv2.imshow('frame', y)
         if cv2.waitKey(1) & 0xFF == ord('q'):
             break

     else:
         break

cap.release()
cv2.destroyAllWindows()

