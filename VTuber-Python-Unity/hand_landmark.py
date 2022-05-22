"""
For finding the face and face landmarks for further manipulication
"""

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


class HandMeshDetector:
    def __init__(self,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5,
                 model_complexity=0)->None:


        self.hands_mesh = mp.solutions.hands.Hands(
                            model_complexity=model_complexity,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        pass

    def findHandMesh(self, img, draw=True):
        # convert the img from BRG to RGB
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        img.flags.writeable = False
        self.results = self.hands_mesh.process(img)

        # Draw the face mesh annotations on the image.
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        self.imgH, self.imgW, self.imgC = img.shape

        self.hands = []
        self.hands_3d = []
        self.handedness = []


        # pose_detector = PoseDetector()

        # img_res, landmarks, wordlist_body = pose_detector.findPose(img)

        worldlist = None

        if self.results.multi_hand_world_landmarks:
            worldlist = [[[hand_point.x, hand_point.y, hand_point.z] for idx, hand_point in enumerate(world_landmarks.landmark)] for world_landmarks in self.results.multi_hand_world_landmarks]

        if self.results.multi_hand_landmarks:
            i = 0
            for hand_landmarks in self.results.multi_hand_landmarks:
                hand_number = self.results.multi_hand_landmarks.index(hand_landmarks)
                # if draw:
                #     self.mp_drawing.draw_landmarks(
                #         img,
                #         hand_landmarks,
                #         mp_hands.HAND_CONNECTIONS,
                #         mp_drawing_styles.get_default_hand_landmarks_style(),
                #         mp_drawing_styles.get_default_hand_connections_style())

                hand = []
                handedness = []

                for id, lmk in enumerate(hand_landmarks.landmark):
                    # print(lmk)
                    x, y = int(lmk.x * self.imgW), int(lmk.y * self.imgH)
                    hand.append([x, y])
                    # show the id of each point on the image
                    # cv2.putText(img, str(id), (x-4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                for id, label in enumerate(self.results.multi_handedness[hand_number].classification):
                    handedness.append(label.label)


                self.hands.append(hand)
                self.handedness.append(label.label)
                # print(self.results.multi_handedness[0].classification.label)

        return img, self.hands, self.handedness, np.array(worldlist)



class PoseDetector():

    def __init__(self,min_detection_confidence=0.5, min_tracking_confidence=0.5)->None:

        self.body_pose = mp_pose.Pose(
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        pass

    def findPose(self, img, draw=True):
        # convert the img from BRG to RGB
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        img.flags.writeable = False
        self.results = self.body_pose.process(img)

        # Draw the face mesh annotations on the image.
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        worldlist = None
        landmarks =None

        if self.results.pose_world_landmarks:
            worldlist = [[body_point.x, body_point.y, body_point.z] for idx, body_point in enumerate(self.results.pose_world_landmarks.landmark)]
        

        if self.results.pose_landmarks:
            landmarks = [[body_point.x, body_point.y] for idx, body_point in enumerate(self.results.pose_landmarks.landmark)]
        
        # if draw:
        #     self.mp_drawing.draw_landmarks(
        #     img,
        #     self.results.pose_landmarks,
        #     mp_pose.POSE_CONNECTIONS,
        #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


        return img, np.array(landmarks), np.array(worldlist)


# sample run of the module
def main():

    hand_detector = HandMeshDetector()
    pose_detector = PoseDetector()


    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        img_res, hands, handed_ness, worldlist_hands = hand_detector.findHandMesh(img)
        img_res, landmarks, wordlist_body =pose_detector.findPose(img_res)

        cv2.imshow('MediaPipe HandMesh', img_res)

        # press "q" to leave
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    # demo code
    main()


