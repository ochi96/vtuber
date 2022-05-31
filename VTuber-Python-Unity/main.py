"""
Main program to run the detection
"""

from argparse import ArgumentParser
import cv2
import mediapipe as mp
import numpy as np
import os

# for TCP connection with unity
import socket
from collections import deque
from platform import system

# face detection and facial landmark
from all_landmarks import FaceMeshDetector, HandsDetector, PoseDetector

# pose estimation and stablization
from pose_estimator2 import PoseEstimator
from pose_estimator import PoseEstimator_
from stabilizer import Stabilizer

# Miscellaneous detections (eyes/ mouth...)
from facial_features import FacialFeatures, Eyes

from process_landmarks import process_body, process_face, process_hands
# global variable
port = 5066         # have to be same as unity


# init TCP connection with unity
# return the socket connected
def init_TCP():
    # address = ("127.0.0.1", port)
    address = ('192.168.0.102', port)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # print(socket.gethostbyname(socket.gethostname()))
    s.connect(address)
    print('unity connected')
    return s

def send_info_to_unity(s, args):
    msg = '%.4f ' * len(args) % args
    s.send(bytes(msg, "utf-8"))

def print_debug_msg(args):
    msg = '%.4f ' * len(args) % args
    print(msg)

def main():

    cap = cv2.VideoCapture(args.cam)

    # Facemesh
    face_detector = FaceMeshDetector()
    hand_detector = HandsDetector()
    pose_detector = PoseDetector()

    # get a sample frame for pose estimation img
    success, img = cap.read()

    # Pose estimation related
    pose_estimator = PoseEstimator((img.shape[0], img.shape[1]))
    pose_estimator_ = PoseEstimator_((img.shape[0], img.shape[1]))

    # print(pose_estimator.model_points_full.shape[0])

    # Initialize TCP connection
    if args.connect:
        socket = init_TCP()

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose

        # first two steps
        img2 = img.copy()
        img3 = img.copy()


        img_facemesh, faces = face_detector.findFaceMesh(img)
        img_hands, hands, handed_ness, worldlist_hands = hand_detector.findHands(img2)
        img_body, landmarks_body, wordlist_body = pose_detector.findPose(img3)

        # print(faces)
        # flip the input image so that it matches the facemesh stuff
        img = cv2.flip(img, 1)

        # if there is any face detected
        if faces:
            face_info = process_face(img_facemesh,  faces, pose_estimator_)
            if args.connect:
                send_info_to_unity(socket,face_info)
            if args.debug:
                print('face_info',face_info)
                pass
            pose_estimator_ = PoseEstimator_((img.shape[0], img.shape[1]))


        if hands:
            hands_info = process_hands(img_hands, hands, handed_ness, worldlist_hands, pose_estimator)
            if args.connect:
                send_info_to_unity(socket, hands_info)
            if args.debug:
                print('hands_info:', hands_info)
                pass
            pose_estimator = PoseEstimator((img.shape[0], img.shape[1]))
        
        if landmarks_body.any():
            body_info = process_body(img_body, landmarks_body, wordlist_body, pose_estimator)
            # cv2.imshow('body',img_body)
            if args.connect:
                send_info_to_unity(socket,body_info)
            if args.debug:
                print('body_info', body_info)
                # cv2.imshow('body', img_body)
                pass
            pose_estimator = PoseEstimator((img.shape[0], img.shape[1]))
        
        else:

            pose_estimator = PoseEstimator((img.shape[0], img.shape[1]))
            pose_estimator_ = PoseEstimator_((img.shape[0], img.shape[1]))
        
          # cv2.imshow('body landmark', img_facemesh)
        horizontal = np.concatenate((img_body, img_facemesh, img_hands), axis=1)
        cv2.imshow('all landmarks', horizontal)
   
        # press "q" to leave
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--cam", type=int,
                        help="specify the camera number if you have multiple cameras",
                        default=0)

    parser.add_argument("--connect", action="store_true",
                        help="connect to unity character",
                        default=False)

    parser.add_argument("--debug", action="store_true",
                        help="showing the camera's image for debugging",
                        default=False)
    args = parser.parse_args()

    # demo code
    main()
