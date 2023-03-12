from flask import Flask, render_template, request, Response
import pickle
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import warnings
from functions import *

app = Flask(__name__)
with open('classification_model\model_pickle', 'rb') as fr:
    model = pickle.load(fr)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

pose_list = ['Downward Dog Pose', 'Goddess Pose', 'Plank Pose', 'Tree Pose', 'Warrior-II Pose']

correct = ''
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)
        
        frame_height, frame_width, _ = frame.shape
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
        
        func_return = get_landmarks(frame, pose)
        if func_return:
            frame, landmarks = func_return
            features = get_features(landmarks)
            if None not in features:
                features = np.array(features)
                features = features[np.newaxis, :]
                y = tf.nn.softmax(model.predict(features, verbose=0)).numpy()
                pose_num = np.argmax(y)
                prediction = pose_list[pose_num]
                if prediction == correct:
                    output = 'Correct'
                else:
                    output = 'Incorrect'
                cv2.putText(frame, '{}'.format(output), (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255), 3)

        ret,buffer=cv2.imencode('.jpg',frame)
        if ret:
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/downward-dog')
def align_dog():
    correct = 'Downward Dog Pose'
    return render_template('downward-dog.html')

@app.route('/warrior')
def align_warrior():
    correct = 'Warrior-II Pose'
    return render_template('warrior.html')

@app.route('/goddess')
def align_goddess():
    correct = 'Goddess Pose'
    return render_template('goddess.html')

@app.route('/plank')
def align_plank():
    correct = 'Plank Pose'
    return render_template('plank.html')

@app.route('/tree')
def align_tree():
    correct = 'Tree Pose'
    return render_template('tree.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)