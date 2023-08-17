#we are including necessary libraries.
import cv2,time,os
from ultralytics import YOLO

#We are running this so that When the RAM started to exceed its capacity ,the system isn't turned off.
os.environ['KMP_DUPLICATE_LIB_OK']='True'

cap=cv2.VideoCapture("Road traffic video for object recognition.mp4") # We will define to use video.
model=YOLO("yolov8x.pt") #We included Yolov8x.pt.

#we defined some of variables.
prev_frame_time=0 #it is about fps.
width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
count_in=set() #we defined the set variables.
count_out=set() 

while True:
    ret,frame=cap.read() #We read incoming image.

    if ret==False: #If Video is Finished , shutdown the system.
        break
    try: #we are calculating FPS(Frame Per Second)
     new_frame_time=time.time()
     fps=1/(new_frame_time-prev_frame_time)
     prev_frame_time=new_frame_time
    except:
       pass
    
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #We transformed Image To RGB .

    cv2.line(frame,(0,int(height/2)),(int(width),int(height/2)),(0,255,0),5,cv2.LINE_8) #to divide equally
        
    result=model.track(frame,persist=True,verbose=False) #We use the model
    #print(result)

    for i in range(len(result[0].boxes)):
      # Loop will be work each one object.
      # Detected location of objects.
      # x1 ve y1 left up corner x2 ve y2 right down corner coordinates.
      x1,y1,x2,y2=result[0].boxes.xyxy[i] 
      score=result[0].boxes.conf[i] 
      cls=result[0].boxes.cls[i]
      ids=result[0].boxes.id[i]

      x1,y1,x2,y2,score,cls,ids=int(x1),int(y1),int(x2),int(y2),float(score),int(cls),int(ids)
      if score<0.5: #if score is less value than threshold , go to initial of the for loop.
            continue
      
      if cls in [1,2,3,5,7]:
          #cv2.putText(frame,f"%.2f"%score,(x1,y1-20),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,2,(255,0,255),2)
          cx=int((x1+x2)/2)
          cy=int((y1+y2)/2)
          
          if cy<int(height/2)and cx<int(width/2):
              cv2.circle(frame,(cx,cy),2,(255,0,0),-1) #we drawed middle of each detected car.
              count_out.add(ids) #we increased concerned variable one when system is detected the car

          elif cy>int(height/2) and cx>int(width/2):
              cv2.circle(frame,(cx,cy),2,(0,0,255),-1) #we drawed middle of each detected car.
              count_in.add(ids)  #we increased concerned variable one when system is detected the car.

    cv2.putText(frame,f"In:{len(count_out)}",(5,25),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2) 
    cv2.putText(frame,f"Out:{len(count_in)}",(5,int(height-25)),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    cv2.putText(frame,f"FPS=%.2f"%fps,(int(width-180),25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)     
          
    cv2.imshow("Frame",frame) #we showed each frame.

    if cv2.waitKey(1) & 0xFF==ord('q'): #if you pressed 'q' key , shutdown the system.
        break

cap.release()
cv2.destroyAllWindows() #to close all the windows.
