
from process_landmarks import process_body, process_face, process_hands
import cv2

def lol(img_hands, hands, handed_ness, worldlist_hands, pose_estimator):
    process_hands(img_hands, hands, handed_ness, worldlist_hands, pose_estimator)
    cv2.imshow('all landmarks', img_hands)
    
    pass
