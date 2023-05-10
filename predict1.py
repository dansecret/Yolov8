from ultralytics import YOLO
import cv2


cap = cv2.VideoCapture(0)

model = YOLO('yolov8m-seg-custom.pt')

results = model(cap)

for result in results:
    for j, mask in result.mask.data:
        
        mask = mask.numpy() + 255
        
        mask = cv2.resize(mask,)
        
        cv2.imwrite(cap)