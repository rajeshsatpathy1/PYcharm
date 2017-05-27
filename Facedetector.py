import numpy as np #Libraries needed for importing arrays
import cv2  #Library of openCV

face_cascade = cv2.CascadeClassifier('E:/MY college files/4th sem/Face recognizer/haarcascade_frontalface_default.xml') #Give your absolute pathname for frontalface xml file
eye_cascade = cv2.CascadeClassifier('E:/MY college files/4th sem/Face recognizer/haarcascade_eye.xml')  #Give your absolute pathname for frontalface xml file
#eye_glasses_cascade = cv2.CascadeClassifier('E:/MY college files/4th sem/Face recognizer/haarcascade_eye_tree_eyeglasses.xml')
#upper_body = cv2.CascadeClassifier('E:/MY college files/4th sem/Face recognizer/haarcascade_upperbody.xml')
cap = cv2.VideoCapture(0)   #start videocapturing from primary laptop webcam - 0

while 1:
    ret, img = cap.read()   #Consider each frame as an image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #convert to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #taking grayscale as reference and usin face_cascade xml

    for (x, y, w, h) in faces:  #(x,y) - position, (w,h) - width and height of rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (255,255, 0), 2)
        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)   #Eyes are checked only after face is detected
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv2.rectangle(roi_gray, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        '''eyeglasses = eye_glasses_cascade.detectMultiScale(roi_gray)
        for (ex1, ey1, ew1, eh1) in eyeglasses:
            cv2.rectangle(roi_color, (ex1, ey1), (ex1 + ew1, ey1 + eh1), (0, 0, 0), 2)'''


    cv2.imshow('img', img)  #Show the original image
    cv2.imshow('gray', gray)    #Show the grayscale image
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()