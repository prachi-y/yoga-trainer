import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import warnings

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def get_landmarks(image, pose):
    landmarks_list = []
    if image.ndim == 3:
        if image.shape[-1] == 3:
            output_image = image.copy()
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(imageRGB)
            height, width, _ = image.shape

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
                for landmark in results.pose_landmarks.landmark:
                    landmarks_list.append((int(landmark.x*width), int(landmark.y*height), (landmark.z*width)))

                return output_image, landmarks_list

            else:
                return
    return


def get_angle(landmark1, landmark2, landmark3):
    warnings.filterwarnings('error')

    pt1 = np.array(landmark1[:1])
    pt2 = np.array(landmark2[:1])
    pt3 = np.array(landmark3[:1])

    line21 = pt1 - pt2
    line23 = pt3 - pt2

    try:
        cosine_angle = np.dot(line21, line23) / (np.linalg.norm(line21) * np.linalg.norm(line23))
        angle_rad = np.arccos(cosine_angle)
        angle_deg = np.degrees(angle_rad)
        
    except:
        return
    
    if angle_deg < 0:
        return (360 + angle_deg)
    
    else: 
        return angle_deg
    
def get_features(landmarks_list):
    shoulder_ang_L = get_angle(landmarks_list[13], landmarks_list[11], landmarks_list[23])
    shoulder_ang_R = get_angle(landmarks_list[14], landmarks_list[12], landmarks_list[24])
    elbow_ang_L = get_angle(landmarks_list[11], landmarks_list[13], landmarks_list[15])
    elbow_ang_R = get_angle(landmarks_list[12], landmarks_list[14], landmarks_list[16])
    waist_ang_L = get_angle(landmarks_list[11], landmarks_list[23], landmarks_list[25])
    waist_ang_R = get_angle(landmarks_list[12], landmarks_list[24], landmarks_list[26])
    knee_ang_L = get_angle(landmarks_list[23], landmarks_list[25], landmarks_list[27])
    knee_ang_R = get_angle(landmarks_list[24], landmarks_list[26], landmarks_list[28])
    
    features = [shoulder_ang_L, shoulder_ang_R, elbow_ang_L, elbow_ang_R,
                waist_ang_L, waist_ang_R, knee_ang_L, knee_ang_R]
    
    features = np.array(features)
    return features