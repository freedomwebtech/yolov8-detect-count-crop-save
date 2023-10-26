import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
from datetime import datetime
import os
now = datetime.now()

model=YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('peoplecount.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)
count=0
tracker=Tracker()   
area=[(54,436),(41,449),(317,494),(317,470)]
area_c=set()

def imgwrite(img):
    now = datetime.now()
    current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    filename = '%s.png' % current_time
    cv2.imwrite(os.path.join(r"C:\Users\freed\Downloads\yolov8parkinglot-main\yolov8parkinglot-main\img",filename), img)
while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue


    frame=cv2.resize(frame,(1020,500))

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c:
            list.append([x1,y1,x2,y2])
            
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
        x3,y3,x4,y4,id=bbox
        results=cv2.pointPolygonTest(np.array(area,np.int32),((x4,y4)),False)
        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
        cv2.circle(frame,(x4,y4),4,(255,0,255),-1)
        cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
        if results>=0:
            crop=frame[y3:y4,x3:x4]
            imgwrite(crop)
#            cv2.imshow(str(id),crop) 
            area_c.add(id)
    cv2.polylines(frame,[np.array(area,np.int32)],True,(255,0,0),2)
    print(area_c)
    k=len(area_c)
    cv2.putText(frame,str(k),(50,60),cv2.FONT_HERSHEY_PLAIN,5,(255,0,0),3)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

