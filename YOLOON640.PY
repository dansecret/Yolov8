import cv2
import time
import numpy as np
import onnxruntime

# Load the ONNX model
onnx_model_path = 'best.onnx'
ort_session = onnxruntime.InferenceSession(onnx_model_path)

cap = cv2.VideoCapture(0)
frameRate = 0
rate = 10.0
prev_frame_time = 0
new_frame_time = 0
framecount = 0
fps = 0
fpsLimit = 1  # throttle limit
startTime = time.time()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Preprocess the frame if needed (resize, normalize, etc.)
        # ...

        # Convert the frame to a tensor
        input_data = np.expand_dims(frame.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0

        # Run inference using ONNX model
        ort_inputs = {ort_session.get_inputs()[0].name: input_data}
        ort_outputs = ort_session.run(None, ort_inputs)

        # Post-process the output if needed
        # ...

        # Display the annotated frame
        annotated_frame = frame  # Replace with your post-processed frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
