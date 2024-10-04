import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("C:/Users/91883/Downloads/yolov3.weights", "C:/Users/91883/Desktop/yolo/yolo.cfg")

# Load COCO labels
classes = []
with open("C:/Users/91883/Downloads/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get layer names and output layers
layer_names = net.getLayerNames()

# For OpenCV versions where `net.getUnconnectedOutLayers()` returns scalar, adjust indexing
try:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]  # For older versions
except:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # For newer versions

# Load the video
video_path = "C:/Users/91883/Downloads/3691658-hd_1920_1080_30fps.mp4" # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw boxes and labels
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display the resulting frame
    cv2.imshow("Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
