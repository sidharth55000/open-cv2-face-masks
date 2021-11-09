import cv2
import numpy as np

face=cv2.CascadeClassifier('/Users/sidhu/cprograms/programspy/cascades/data/haarcascade_frontalface_alt2.xml')#cascade classifier location
img=cv2.imread('bo.jpg')
img_=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#hori = np.hstack((img, img_))
def empty():
    pass

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV",640,240)
cv2.createTrackbar("HUE Min","HSV",0,179,empty)
cv2.createTrackbar("HUE Max","HSV",179,179,empty)
cv2.createTrackbar("SAT Min","HSV",0,255,empty)
cv2.createTrackbar("SAT Max","HSV",255,255,empty)
cv2.createTrackbar("VALUE Min","HSV",0,255,empty)
cv2.createTrackbar("VALUE Max","HSV",255,255,empty)

while True:
    #cv2.imshow('',img)
    #cv2.imshow('',img_)
    #cv2.imshow('',hori)
    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")
    print(h_min)

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(img_,lower,upper)
    result = cv2.bitwise_and(img,img, mask = mask)



    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    hStack = np.hstack([img, mask, result])
    imgs=cv2.resize(hStack,(960,540))
    # cv2.imshow('Original', img)
    # cv2.imshow('HSV Color Space', imgHsv)
    # cv2.imshow('Mask', mask)
    # cv2.imshow('Result', result)
    cv2.imshow('Horizontal Stacking', imgs)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.DestroyAllWindowa()
