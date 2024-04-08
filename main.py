from ultralytics import YOLO
import cv2
import math

# Load the YOLO model specified in the 'fire.pt' file
model = YOLO('best.pt')

# Define class names (in this case, just 'FIRE')
classnames = ['fire']

# Open the laptop's camera (use 0 for the default camera, or specify the camera index)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Resize the frame (optional, adjust the size as needed)
    frame = cv2.resize(frame, (640, 480))

    # Perform object detection with the YOLO model
    result = model(frame, stream=True)

    # Getting bbox, confidence, and class names information to work with
    for info in result:
        # Retrieve the detected bounding boxes for each object in the current frame
        boxes = info.boxes
        for box in boxes:
            # Retrieve the confidence (detection probability) associated with the bounding box
            confidence = box.conf[0]
            # Multiply confidence by 100 and round up to get a percentage
            confidence = math.ceil(confidence * 100)
            # Retrieve the predicted class of the detected object (in this case, there's only one class, "FIRE")
            Class = int(box.cls[0])
            if 0 <= Class < len(classnames):
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cv2.putText(frame, f'{classnames[Class]} {confidence}%', (x1 + 8, y1 + 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            else:
                print(f"Invalid class index: {Class}")

    # Display the frame with detected objects
    cv2.imshow('Fire Detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
