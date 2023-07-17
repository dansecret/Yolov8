import cv2
import time
import numpy as np
from ultralytics import YOLO

# Set the device to GPU
import torch
torch.device('cuda')
# Load the YOLOv8 model
model = YOLO('yolov8m-seg-custom.pt')

cap = cv2.VideoCapture(0)
frameRate = 0
rate = 10.0
prev_frame_time = 0
new_frame_time = 0
framecount = 0
fps = 0
fpsLimit = 1# throttle limit
startTime = time.time()
# reset time
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        for result in results:
            boxes = result.boxes.cpu().numpy() 
            for box in boxes: 
                r = box.xyxy[0].astype(int)
                cv2.rectangle(frame, r[:2], r[2:], (255, 0, 255), 2)
                print("suuuuuuuuuuuuuuu")
                print(r[:2][0], r[:2][1])
                cv2.putText(annotated_frame,str(r[:2][0])+","+str(r[:2][1]), (r[:2][0]+5,r[:2][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2, cv2.LINE_AA)
        # Display the annotated frame
        # fps = int(cap.get(cv2.CAP_PROP_FPS))
        nowTime = time.time()
        frameRate += rate
        times = int(nowTime - startTime)
        if times > fpsLimit:
            fps = frameRate/times
            startTime = time.time()
            frameRate = 0
        cv2.putText(annotated_frame,str(fps),(50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 0, 0),2,cv2.LINE_AA)
        cv2.imshow("YOLOv8 Inference",annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
