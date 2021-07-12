##Necessary Imports
import cv2
from tracker import *

#Tracker Calling
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture('highway.mp4')


object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=45)


while cap.isOpened():

    ret,frame = cap.read() #Extracting frame from video

    roi = frame[350:650, 570:780] #Selecting ROI area

    mask = object_detector.apply(roi)

    _,mask = cv2.threshold(mask,254,255,cv2.THRESH_BINARY) #Image binarization



    contours,_ = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #Finding contour points
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>150 :

            #cv2.drawContours(roi,[cnt], -1 ,(0,255,0),1)
            x,y,w,h = cv2.boundingRect(cnt)
            print("X: ",w," Y: ",h)
            # if x>150 and y>150:
            #     cv2.rectangle(roi,(x,y),(x+w,y+h),(0,150,0),1)
            #     detections.append([x,y,w,h])
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 150, 0), 1)
            detections.append([x,y,w,h])
            #print(x,y,w,h)

    boxes_ids = tracker.update(detections)

    #print(boxes_ids)
    for box_id in boxes_ids:

        x,y,w,h,id = box_id
        print(x)
        print(id)
        if w<38:    #Detecting Bike
            name="bike"
            cv2.putText(roi,name,(x,y-15),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)

            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 150, 0), 1)


        elif w>=38:      #Detecting Cars

            name="car"
            cv2.putText(roi,name,(x,y-15),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)

            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 150, 0), 2)





    cv2.putText(frame,"Total vehicles:"+str(id),(20,50),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)




    cv2.imshow("VideoFeed",frame)

    key = cv2.waitKey(30)
    if key == ord('n') or key == ord('p'):
        break



cap.release()
cap.destroyAllWindows() #closing all windoes