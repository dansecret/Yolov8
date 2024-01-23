import cv2
import torch
import numpy as np
from torch2trt import TRTModule
from ultralytics import YOLO

# Load the pre-converted TensorRT engine in INT8 mode
engine_path = "best.engine"
model_trt_int8 = TRTModule()
model_trt_int8.load_state_dict(torch.load(engine_path))

cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Preprocess the frame if needed

        # Run inference using the TensorRT engine in INT8 mode
        input_tensor = torch.from_numpy(frame).cuda().float()
        results = model_trt_int8(input_tensor[None, ...])

        # Process the results and annotate the frame
        # ...

        # Display the annotated frame
        cv2.imshow("Inference with INT8 TensorRT", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
