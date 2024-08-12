import time
from flask import Flask, render_template, request, redirect, url_for, session, flash
import cv2
import dlib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import webview
import pyttsx3
import threading

app = Flask(__name__)
window = webview.create_window('Attendance System', app)
app.config['UPLOAD_FOLDER'] = 'dataset'
app.secret_key = '9876'
PIN = '9876'  # Set your PIN here+

# Initialize Dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Global dictionary to hold face encodings
face_encodings_dict = {}

def initialize_employee_file():
    if not os.path.exists('employees.xlsx'):
        df = pd.DataFrame(columns=['Name', 'Date', 'EntryTime', 'ExitTime'])
        df.to_excel('employees.xlsx', index=False)

def mark_attendance(name, event):
    df = pd.read_excel('employees.xlsx')
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    if event == 'entry':
        existing_entry = df[(df['Name'] == name) & (df['Date'] == current_date) & (df['EntryTime'] != '')]
        if not existing_entry.empty:
            return False  # Indicate that the attendance is already marked
        new_entry = pd.DataFrame([[name, current_date, current_time, '']],
                                 columns=['Name', 'Date', 'EntryTime', 'ExitTime'])
        df = pd.concat([df, new_entry], ignore_index=True)
    elif event == 'exit':
        latest_entry_idx = df[(df['Name'] == name) & (df['Date'] == current_date) & (
                df['ExitTime'].isna() | (df['ExitTime'] == ''))].index.max()

        if pd.notna(latest_entry_idx):
            df.at[latest_entry_idx, 'ExitTime'] = current_time

    df.to_excel('employees.xlsx', index=False)

def load_known_faces():
    global face_encodings_dict
    face_encodings_dict.clear()  # Clear the dictionary before loading known faces
    for user_dir in os.listdir(app.config['UPLOAD_FOLDER']):
        user_path = os.path.join(app.config['UPLOAD_FOLDER'], user_dir)
        if os.path.isdir(user_path):
            for file_name in os.listdir(user_path):
                if file_name.endswith('.jpg'):
                    image_path = os.path.join(user_path, file_name)
                    image = cv2.imread(image_path)
                    encoding = get_face_encoding(image)
                    if encoding is not None:
                        face_encodings_dict[user_dir] = encoding  # Store the encoding with the username as the key

def get_face_encoding(image):
    detections = detector(image, 1)
    if len(detections) > 0:
        shape = sp(image, detections[0])
        face_descriptor = facerec.compute_face_descriptor(image, shape)
        return np.array(face_descriptor)
    return None

load_known_faces()


def play_video_and_capture_images(video_path, user_dir, name, window_size):
    cap = cv2.VideoCapture(video_path)
    video_capture = cv2.VideoCapture(0)

    count = 0
    start_time = time.time()

    while cap.isOpened() and count < 15 and (time.time() - start_time) < 10:
        ret, video_frame = cap.read()
        ret2, frame = video_capture.read()

        if not ret or not ret2:
            break

        video_frame = cv2.resize(video_frame, window_size)
        cv2.imshow('Video', video_frame)

        # Hide webcam window
        cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Webcam', 0, 0)
        # cv2.imshow('Webcam', frame)

        file_path = os.path.join(user_dir, f"{name}_{count}.jpg")
        cv2.imwrite(file_path, frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        while (time.time() - start_time) < count:
            time.sleep(0.1)

    cap.release()
    video_capture.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        pin = request.form['pin']
        if pin == PIN:
            session['authenticated'] = True
            return redirect(url_for('register'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'authenticated' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        name = request.form['name']

        # Create a directory for the user if it doesn't exist
        user_dir = os.path.join(app.config['UPLOAD_FOLDER'], name)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)

        # Get the WebView window size
        window_size = (window.width, window.height)

        # Start the video playback and image capture
        play_video_and_capture_images('WhatsApp Video 2024-07-30 at 17.23.18_a74594ff.mp4', user_dir, name, window_size)

        return redirect(url_for('index'))
    return render_template('register.html')


@app.route('/attendance', methods=['GET', 'POST'])
def attendance():

    recognized_names = []

    if request.method == 'POST':
        event = request.form['event']
        video_capture = cv2.VideoCapture(0)
        unknown_detected = True

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector(rgb_frame, 1)

            for d in detections:
                shape = sp(rgb_frame, d)
                face_descriptor = facerec.compute_face_descriptor(rgb_frame, shape)
                face_encoding = np.array(face_descriptor)

                for name, known_encoding in face_encodings_dict.items():
                    matches = np.linalg.norm(known_encoding - face_encoding)
                    if matches < 0.6:  # You can adjust the threshold as needed
                        recognized_names.append(name)
                        mark_attendance(name, event)
                        engine = pyttsx3.init()
                        engine.say(f"{name}, your attendance is marked successfully")
                        engine.runAndWait()
                        unknown_detected = False
                        break # Exit the loop after marking attendance

            if recognized_names or unknown_detected:
                break  # Exit the loop if a face is recognized and attendance is marked

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

        if unknown_detected:
            engine = pyttsx3.init()
            engine.say("Sorry!!Please try again")
            engine.runAndWait()
            recognized_names.append("Unknown person detected")

        return render_template('index.html', names=recognized_names)
    return render_template('index.html', names=[])


if __name__ == '__main__':
    initialize_employee_file()
    webview.start()