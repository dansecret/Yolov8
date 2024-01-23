import cv2
import time
import numpy as np
import onnxruntime

# Load the ONNX model
onnx_model_path = "best.onnx"
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
        # Preprocess the frame
        input_frame = cv2.resize(frame, (640, 640))
        input_data = np.expand_dims(input_frame.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0

        # Run inference using ONNX model
        ort_inputs = {ort_session.get_inputs()[0].name: input_data}
        ort_outputs = ort_session.run(None, ort_inputs)

        # Post-process the output
        detected_boxes = ort_outputs[0][0]
        for box in detected_boxes:
            if box[4] > 0.5:  # Confidence threshold
                x, y, w, h = (box[:4] * np.array([640, 640, 640, 640])).astype(int)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        nowTime = time.time()
        frameRate += rate
        times = int(nowTime - startTime)
        if times > fpsLimit:
            fps = frameRate/times
            startTime = time.time()
            frameRate = 0

        # Display the annotated frame
        cv2.putText(frame, str(fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("YOLOv8 Inference", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
