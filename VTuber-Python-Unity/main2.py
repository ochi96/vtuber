"""
Main program to run the detection
"""

from argparse import ArgumentParser
import cv2
import mediapipe as mp
import numpy as np

# for TCP connection with unity
import socket
from collections import deque
from platform import system

# face detection and facial landmark
from all_landmarks import FaceMeshDetector, HandsDetector, PoseDetector

# pose estimation and stablization
from pose_estimator2 import PoseEstimator
from stabilizer import Stabilizer

# Miscellaneous detections (eyes/ mouth...)
from facial_features import FacialFeatures, Eyes

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
    msg = args
    print(msg)

def main():
    # Handmesh
    detector = HandsDetector()

    image_points = np.zeros((21, 2))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    # Initialize TCP connection
    if args.connect:
        socket = init_TCP()




    cap = cv2.VideoCapture(args.cam)
    while cap.isOpened():
        success, img = cap.read()

        pose_estimator = PoseEstimator((img.shape[0], img.shape[1]))

        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose

        # first two steps
        img_handmesh, hands, handed_ness, world_list = detector.findHands(img)

        # world_list = [world_list]

        print('handedness: ', handed_ness)


        print("Number of hands: ", len(handed_ness) )
        no_of_hands = len(handed_ness)
        # flip the input image so that it matches the facemesh stuff
        img = cv2.flip(img, 1)



        # if there is any hand detected
        if hands:
            all_hands = []
            for i in range(no_of_hands):
                # get all hands
                for j in range(len(image_points)):
                    image_points[j, 0] = hands[i][j][0]
                    image_points[j, 1] = hands[i][j][1]
                
                interesting_points = np.array([image_points[i] for i in [0, 1, 5, 9, 13, 17]])
                world_coordinates = np.array([world_list[i][j] for j in [0, 1, 5, 9, 13, 17]])
                # The third step: pose estimation
                # pose: [[rvec], [tvec]]
                pose = pose_estimator.solve_pose_by_all_points(interesting_points, world_coordinates)

                # Stabilize the pose.
                steady_pose = []
                pose_np = np.array(pose).flatten()

                for value, ps_stb in zip(pose_np, pose_stabilizers):
                    ps_stb.update([value])
                    steady_pose.append(ps_stb.state[0])

                steady_pose = np.reshape(steady_pose, (-1, 3))

                # calculate the roll/ pitch/ yaw
                # roll: +ve when the axis pointing upward
                # pitch: +ve when we look upward
                # yaw: +ve when we look left
                roll = np.clip(np.degrees(steady_pose[0][1]), -90, 90)
                pitch = np.clip(-(180 + np.degrees(steady_pose[0][0])), -90, 90)
                yaw =  np.clip(np.degrees(steady_pose[0][2]), -90, 90)

                all_hands.append(([roll, pitch, yaw], handed_ness[i]))

                # print(steady_pose)

                pose_estimator.draw_axes(img_handmesh, steady_pose[0], steady_pose[1])


            if args.debug:
                print_debug_msg(all_hands)


            # pose_estimator.draw_annotation_box(img_handmesh, pose[0], pose[1], color=(255, 128, 128))

            # pose_estimator.draw_axis(img, pose[0], pose[1])

            # pose_estimator.draw_axes(img_handmesh, steady_pose[0], steady_pose[1])

        else:
            # reset our pose estimator
            pose_estimator = PoseEstimator((img_handmesh.shape[0], img_handmesh.shape[1]))

        if args.debug:
            cv2.imshow('Hands landmarks', img_handmesh)

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
