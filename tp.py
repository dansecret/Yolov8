import cv2
import torch
from torch2trt import TRTModule

# Load the pre-converted TensorRT engine
engine_path = "best.engine"
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(engine_path))

cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Preprocess the frame if needed

        # Run inference using the TensorRT engine
        input_tensor = torch.from_numpy(frame).cuda().float()
        results = model_trt(input_tensor[None, ...])

        # Process the results and annotate the frame
        # ...

        # Display the annotated frame
        cv2.imshow("Inference with TensorRT", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
