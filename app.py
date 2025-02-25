import cv2
import numpy as np
import tensorflow.lite as tflite

# Load TFLite object detection model
od_interpreter = tflite.Interpreter(model_path="detect.tflite")
od_interpreter.allocate_tensors()

# Get input and output tensors details
od_input_details = od_interpreter.get_input_details()
od_output_details = od_interpreter.get_output_details()
od_img_width, od_img_height = od_input_details[0]['shape'][1], od_input_details[0]['shape'][2]

# Load TFLite classification model
clf_interpreter = tflite.Interpreter(model_path="apple_classifier.tflite")
clf_interpreter.allocate_tensors()

# Get input and output tensors details
clf_input_details = clf_interpreter.get_input_details()
clf_output_details = clf_interpreter.get_output_details()
clf_img_width, clf_img_height = clf_input_details[0]['shape'][1], clf_input_details[0]['shape'][2]

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w, _ = frame.shape
    resized_frame = cv2.resize(frame, (od_img_width, od_img_height))
    input_data = np.expand_dims(resized_frame.astype(np.float32) / 255.0, axis=0)

    # Run object detection
    od_interpreter.set_tensor(od_input_details[0]['index'], input_data)
    od_interpreter.invoke()
    boxes = od_interpreter.get_tensor(od_output_details[0]['index'])[0]  # Bounding boxes
    classes = od_interpreter.get_tensor(od_output_details[1]['index'])[0]  # Class IDs
    scores = od_interpreter.get_tensor(od_output_details[2]['index'])[0]  # Confidence scores
    
    for i in range(len(scores)):
        if scores[i] > 0.3:  # Confidence threshold
            class_id = int(classes[i])
            if class_id == 0:  # Assuming class 0 is 'apple'
                y_min, x_min, y_max, x_max = boxes[i]
                x, y, x_max, y_max = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)
                
                apple_img = frame[y:y_max, x:x_max]
                apple_img = cv2.resize(apple_img, (clf_img_width, clf_img_height))
                apple_img = apple_img.astype("float32") / 255.0
                apple_img = np.expand_dims(apple_img, axis=0)

                # Run classification model
                clf_interpreter.set_tensor(clf_input_details[0]['index'], apple_img)
                clf_interpreter.invoke()
                prediction = clf_interpreter.get_tensor(clf_output_details[0]['index'])
                
                label = "Ripe Apple" if np.argmax(prediction) == 0 else "Unripe Apple"
                
                cv2.rectangle(frame, (x, y), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Apple Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
