import face_recognition
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

# -------------------- SETTINGS --------------------
# Directory containing known face images
known_faces_dir = '/Users/archita/Desktop/Coding/Projects/Face Recognition Attendance System/known_faces'

# Pick your MacBook webcam index (usually 1 if iPhone is connected)
CAMERA_INDEX = 1

# Output CSV file
OUTPUT_CSV = 'attendance_log.csv'

# -------------------- LOAD KNOWN FACES --------------------
known_face_encodings = []
known_face_names = []

for file_name in os.listdir(known_faces_dir):
    if file_name.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(known_faces_dir, file_name)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(file_name)[0])
        else:
            print(f"[WARNING] No face found in {file_name}, skipping this image.")

if len(known_face_encodings) == 0:
    print("[ERROR] No valid face encodings found. Exiting.")
    exit()

# -------------------- INITIALIZE ATTENDANCE --------------------
attendance_log = {}

# -------------------- INITIALIZE WEBCAM --------------------
video_capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
if not video_capture.isOpened():
    print(f"[ERROR] Cannot access camera with index {CAMERA_INDEX}.")
    exit()
print("[INFO] Camera successfully opened.")

# -------------------- REAL-TIME FACE RECOGNITION --------------------
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[ERROR] Failed to capture frame from camera.")
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and encode them
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the closest known face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw rectangle and name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # Log attendance
        if name != "Unknown" and name not in attendance_log:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            attendance_log[name] = timestamp
            print(f"[LOGGED] {name} at {timestamp}")

    # Display frame
    cv2.imshow('Facial Recognition Attendance', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------- CLEAN UP --------------------
video_capture.release()
cv2.destroyAllWindows()

# Export attendance to CSV
if attendance_log:
    df = pd.DataFrame(list(attendance_log.items()), columns=['Name', 'Timestamp'])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Attendance exported to {OUTPUT_CSV}")
else:
    print("[INFO] No attendance logged.")


### Press 'q' to quit and export the CSV.