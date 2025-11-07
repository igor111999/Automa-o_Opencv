import time

import cv2
import numpy as np
import math
import os

from selenium.webdriver.common.devtools.v85.debugger import pause

cap = cv2.VideoCapture(0)

kernel = np.ones((3,3), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100,100), (300,300), (0,255,0), 0)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.dilate(mask, kernel, iterations=4)


    mask = cv2.GaussianBlur(mask, (5,5), 100)


    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    cnt = max(contours, key=lambda x: cv2.contourArea(x))
    epsilon = 0.0005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    hull = cv2.convexHull(approx, returnPoints=False)
    if hull is None or len(hull) < 3:
        continue

    defects = cv2.convexityDefects(approx, hull)
    if defects is None:
        continue

    l = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(approx[s][0])
        end = tuple(approx[e][0])
        far = tuple(approx[f][0])

        a = math.dist(end, start)
        b = math.dist(far, start)
        c = math.dist(end, far)
        s_area = (a+b+c)/2
        ar = math.sqrt(s_area*(s_area-a)*(s_area-b)*(s_area-c))
        d = (2*ar)/a

        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        if angle <= 90 and d > 30:
            l += 1
            cv2.circle(roi, far, 3, [255,0,0], -1)

        cv2.line(roi, start, end, [0,255,0], 2)

    l += 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    areacnt = cv2.contourArea(cnt)
    areahull = cv2.contourArea(cv2.convexHull(cnt))
    arearatio = ((areahull - areacnt)/areacnt) * 100
    if l == 1:
        if areacnt < 2000:
            cv2.putText(frame, 'esperando dados', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        else:
            if arearatio < 12:
                cv2.putText(frame, '0 = Navegador', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                os.system("start Chrqome.exe")


            elif arearatio < 17.5:
                cv2.putText(frame, 'ok, firefox', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                os.system("start firefox.exe")
            else:
                cv2.putText(frame, '1 = MEDIA PLAYER', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                os.system("start wmplayer.exe")

    elif l == 2:
        cv2.putText(frame, '2 = Excel', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        #os.system('start excel')

    elif l == 3:
        if arearatio < 27:
            cv2.putText(frame, '3 = PowerPoint', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            #os.system("start powerpnt")
        else:
            cv2.putText(frame, '0 = Navegador', (0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

    cv2.imshow('mask', mask)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

