# Face Recognition Attendance System

A Python Streamlit application to take attendance using face recognition snapshots.

## Features

- Snapshot-based webcam attendance (Arrival & Departure)
- Marks On Time / Late automatically
- Stores daily attendance CSV logs
- Visualizations with charts for Arrival & Departure trends
- Contact Me section included

## Technologies Used

- Python
- Streamlit
- OpenCV
- face_recognition
- Pandas
- Plotly
- Pillow (PIL)

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/face-recognition-attendance.git


2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run attendance_system_on_streamlit.py
   ```

## Folder Structure

face-recognition-attendance/
│
├─ known_faces/          # Images of known people
├─ logs/                 # Attendance CSV logs
├─ attendance_system_on_streamlit.py
├─ README.md
├─ requirements.txt
└─ .gitignore

## Author

Archita Thakur
✉️ architath27@gmail.com