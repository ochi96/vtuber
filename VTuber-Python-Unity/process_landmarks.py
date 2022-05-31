import numpy as np
from argparse import ArgumentParser
import cv2

# for TCP connection with unity
from collections import deque
from platform import system


# pose estimation and stablization
from stabilizer import Stabilizer

# Miscellaneous detections (eyes/ mouth...)
from facial_features import FacialFeatures, Eyes

# Introduce scalar stabilizers for pose.
pose_stabilizers = [Stabilizer(
    state_num=2,
    measure_num=1,
    cov_process=0.1,
    cov_measure=0.1) for _ in range(6)]

# for eyes
eyes_stabilizers = [Stabilizer(
    state_num=2,
    measure_num=1,
    cov_process=0.1,
    cov_measure=0.1) for _ in range(6)]

# for mouth_dist
mouth_dist_stabilizer = Stabilizer(
    state_num=2,
    measure_num=1,
    cov_process=0.1,
    cov_measure=0.1
)


def process_face(img_facemesh, faces, pose_estimator_):
    # only get the first face
    image_points = np.zeros((pose_estimator_.model_points_full.shape[0], 2))
    # extra 10 points due to new attention model (in iris detection)
    iris_image_points = np.zeros((10, 2))

    for i in range(len(image_points)):
        image_points[i, 0] = faces[0][i][0]
        image_points[i, 1] = faces[0][i][1]
        
    for j in range(len(iris_image_points)):
        iris_image_points[j, 0] = faces[0][j + 468][0]
        iris_image_points[j, 1] = faces[0][j + 468][1]

    # The third step: pose estimation
    # pose: [[rvec], [tvec]]
    pose = pose_estimator_.solve_pose_by_all_points(image_points)

    x_ratio_left, y_ratio_left = FacialFeatures.detect_iris(image_points, iris_image_points, Eyes.LEFT)
    x_ratio_right, y_ratio_right = FacialFeatures.detect_iris(image_points, iris_image_points, Eyes.RIGHT)


    ear_left = FacialFeatures.eye_aspect_ratio(image_points, Eyes.LEFT)
    ear_right = FacialFeatures.eye_aspect_ratio(image_points, Eyes.RIGHT)

    pose_eye = [ear_left, ear_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right]

    mar = FacialFeatures.mouth_aspect_ratio(image_points)
    mouth_distance = FacialFeatures.mouth_distance(image_points)

    # Stabilize the pose.
    steady_pose = []
    pose_np = np.array(pose).flatten()

    for value, ps_stb in zip(pose_np, pose_stabilizers):
        ps_stb.update([value])
        steady_pose.append(ps_stb.state[0])

    steady_pose = np.reshape(steady_pose, (-1, 3))

    # stabilize the eyes value
    steady_pose_eye = []
    for value, ps_stb in zip(pose_eye, eyes_stabilizers):
        ps_stb.update([value])
        steady_pose_eye.append(ps_stb.state[0])

    mouth_dist_stabilizer.update([mouth_distance])
    steady_mouth_dist = mouth_dist_stabilizer.state[0]

    # print("rvec steady (x, y, z) = (%f, %f, %f): " % (steady_pose[0][0], steady_pose[0][1], steady_pose[0][2]))
    # print("tvec steady (x, y, z) = (%f, %f, %f): " % (steady_pose[1][0], steady_pose[1][1], steady_pose[1][2]))

    # calculate the roll/ pitch/ yaw
    # roll: +ve when the axis pointing upward
    # pitch: +ve when we look upward
    # yaw: +ve when we look left
    roll = np.clip(np.degrees(steady_pose[0][1]), -90, 90)
    pitch = np.clip(-(180 + np.degrees(steady_pose[0][0])), -90, 90)
    yaw =  np.clip(np.degrees(steady_pose[0][2]), -90, 90)

    # print("Roll: %.2f, Pitch: %.2f, Yaw: %.2f" % (roll, pitch, yaw))
    # print("left eye: %.2f, %.2f; right eye %.2f, %.2f"
    #     % (steady_pose_eye[0], steady_pose_eye[1], steady_pose_eye[2], steady_pose_eye[3]))
    # print("EAR_LEFT: %.2f; EAR_RIGHT: %.2f" % (ear_left, ear_right))
    # print("MAR: %.2f; Mouth Distance: %.2f" % (mar, steady_mouth_dist))
    pose_estimator_.draw_axes(img_facemesh, steady_pose[0], steady_pose[1])

    face_info = (roll, pitch, yaw,
            ear_left, ear_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right,
            mar, mouth_distance)
    
    return face_info

def process_hands(img_hands, hands, handed_ness, worldlist_hands, pose_estimator):
    image_points = np.zeros((21, 2))
    print('handedness: ', handed_ness)
    print("Number of hands: ", len(handed_ness) )
    no_of_hands = len(handed_ness)
    hands_info = []
    for i in range(no_of_hands):
        # get all hands
        for j in range(len(image_points)):
            image_points[j, 0] = hands[i][j][0]
            image_points[j, 1] = hands[i][j][1]
        
        interesting_points = np.array([image_points[i] for i in [0, 1, 5, 9, 13, 17]])
        world_coordinates = np.array([worldlist_hands[i][j] for j in [0, 1, 5, 9, 13, 17]])

        fingertip_points = np.array([image_points[i] for i in [4, 8, 12, 16, 20]])
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
        # print(steady_pose)

        roll = np.clip(np.degrees(steady_pose[0][1]), -90, 90)
        pitch = np.clip(-(180 + np.degrees(steady_pose[0][0])), -90, 90)
        yaw =  np.clip(np.degrees(steady_pose[0][2]), -90, 90)

        hands_info.append(([roll, pitch, yaw], handed_ness[i], fingertip_points))
        pose_estimator.draw_axes(img_hands, steady_pose[0], steady_pose[1])
    

    return hands_info


def process_body(img_body, landmarks_body, worldlist_body, pose_estimator):
    image_points = np.zeros((33, 2))
    body_info = []

    for i in range(len(image_points)):
        image_points[i, 0] = landmarks_body[i][0]
        image_points[i, 1] = landmarks_body[i][1]
    
    
    interesting_points = np.array([image_points[i] for i in [5,6, 11, 12, 13, 14]])
    world_coordinates = np.array([worldlist_body[j] for j in [5,6, 11, 12, 13, 14]])
    # interesting_points = np.array(image_points)
    # world_coordinates = np.array(worldlist_body)
    pose = pose_estimator.solve_pose_by_all_points(interesting_points, world_coordinates)

    # Stabilize the pose.
    steady_pose = []
    pose_np = np.array(pose).flatten()

    for value, ps_stb in zip(pose_np, pose_stabilizers):
        ps_stb.update([value])
        steady_pose.append(ps_stb.state[0])

    steady_pose = np.reshape(steady_pose, (-1, 3))

    roll = np.clip(np.degrees(steady_pose[0][1]), -90, 90)
    pitch = np.clip(-(180 + np.degrees(steady_pose[0][0])), -90, 90)
    yaw =  np.clip(np.degrees(steady_pose[0][2]), -90, 90)

    body_info.append(([roll, pitch, yaw]))
    pose_estimator.draw_axes(img_body, steady_pose[0], steady_pose[1])

    return body_info