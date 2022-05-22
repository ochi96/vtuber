
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


class FaceMeshDetector:
    def __init__(self,
                 static_image_mode=False,
                 max_num_faces=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Facemesh
        self.mp_face_mesh = mp.solutions.face_mesh
        # The object to do the stuffs
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            self.static_image_mode,
            self.max_num_faces,
            True,
            self.min_detection_confidence,
            self.min_tracking_confidence
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        # convert the img from BRG to RGB
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        img.flags.writeable = False
        self.results = self.face_mesh.process(img)

        # Draw the face mesh annotations on the image.
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        self.imgH, self.imgW, self.imgC = img.shape

        self.faces = []

        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        image = img,
                        landmark_list = face_landmarks,
                        connections = self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec = self.drawing_spec,
                        connection_drawing_spec = self.drawing_spec)

                face = []
                for id, lmk in enumerate(face_landmarks.landmark):
                    x, y = int(lmk.x * self.imgW), int(lmk.y * self.imgH)
                    face.append([x, y])

                    # show the id of each point on the image
                    # cv2.putText(img, str(id), (x-4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

                self.faces.append(face)

        return img, self.faces


class HandsDetector:
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

    def findHands(self, img, draw=True):
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
        self.handedness = []
        worldlist = None

        if self.results.multi_hand_world_landmarks:
            worldlist = [[[hand_point.x, hand_point.y, hand_point.z] for idx, hand_point in enumerate(world_landmarks.landmark)] for world_landmarks in self.results.multi_hand_world_landmarks]

        if self.results.multi_hand_landmarks:
            i = 0
            for hand_landmarks in self.results.multi_hand_landmarks:
                hand_number = self.results.multi_hand_landmarks.index(hand_landmarks)
                if draw:
                    self.mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                hand = [[int(lmk.x * self.imgW), int(lmk.y * self.imgH)] for id, lmk in enumerate(hand_landmarks.landmark)]
                handedness = [label.label for id, label in enumerate(self.results.multi_handedness[hand_number].classification)]
                self.hands.append(hand)
                self.handedness.append(handedness[0])

        return img, self.hands, self.handedness, np.array(worldlist)



class PoseDetector():

    def __init__(self,min_detection_confidence=0.7, min_tracking_confidence=0.5)->None:

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

        self.imgH, self.imgW, self.imgC = img.shape

        # Draw the face mesh annotations on the image.
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        worldlist = None
        landmarks = None

        if self.results.pose_world_landmarks:
            worldlist = [[body_point.x, body_point.y, body_point.z] for idx, body_point in enumerate(self.results.pose_world_landmarks.landmark)]
        

        if self.results.pose_landmarks:
            landmarks = [[int(body_point.x * self.imgW), int(body_point.y * self.imgW)] for idx, body_point in enumerate(self.results.pose_landmarks.landmark)]
 
        if draw:
            self.mp_drawing.draw_landmarks(
            img,
            self.results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        return img, np.array(landmarks), np.array(worldlist)


# sample run of the module
def main():

    detector = FaceMeshDetector()
    hand_detector = HandsDetector()
    pose_detector = PoseDetector()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        

        img_face, faces = detector.findFaceMesh(img)
        img_hands, hands, handed_ness, worldlist_hands = hand_detector.findHands(img)
        img_pose, landmarks_body, wordlist_body = pose_detector.findPose(img)

        if faces:
            print('faces detected')
        if hands:
            print('hands detected')
        if landmarks_body.any():
            print('body detected')
        
        # cv2.imshow('MediaPipe FaceMesh', img_hands)
        # cv2.imshow('MediaPipe FaceMesh', img_face)

        horizontal = np.concatenate((img_face, img_hands, img_pose), axis=1)
 
        cv2.imshow('all landmarks', horizontal)

        # press "q" to leave
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    # demo code
    main()


