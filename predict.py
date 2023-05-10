from ultralytics import YOLO


model = YOLO('yolov8m-seg-custom.pt')

model.predict(source = 0, show=True,val=True,show_conf=True, save=True, show_labels=True,conf=0.5, save_txt=True, save_crop=False,line_width=2, box=True, visualize=False)

# frame_count = 0
# start_time = time.time()


# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     cv2.imshow("Camera", frame)

#     frame_count += 1

#     # calculate the FPS every 10 frames
#     if frame_count == 10:
#         fps = 10 / (time.time() - start_time)
#         print("FPS:", round(fps, 2))
#         frame_count = 0
#         start_time = time.time()

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # webcame with 0

