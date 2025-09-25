import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from enum import IntEnum
from pose_helpers import is_standing, is_sitting, is_walking, is_raised
from config import OBJECTS_MODEL_PATH, POSE_MODEL_PATH, SCORE_THRESHOLD


class Landmark(IntEnum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


class PeopleDetector:
    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        ObjectDetector = mp.tasks.vision.ObjectDetector
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=OBJECTS_MODEL_PATH),
            max_results=10,
            score_threshold=SCORE_THRESHOLD,
            category_allowlist=["person"],
            running_mode=VisionRunningMode.IMAGE)
        
        self.detector = ObjectDetector.create_from_options(options)

    def detect(self, image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = self.detector.detect(mp_image)

        people = []
        for detection in result.detections:
            people.append(detection.bounding_box)
        return people

    def close(self):
        self.detector.close()

class PoseDetector:
    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE)
        
        self.landmarker = PoseLandmarker.create_from_options(options)

    def detect(self, image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        return self.landmarker.detect(mp_image)

    def close(self):
        self.landmarker.close()

def draw_landmarks_on_image(image, landmark_result): 
    pose_landmarks_list = landmark_result.pose_landmarks
    annotated_image = np.copy(image)
    if len(pose_landmarks_list) == 0:
        return annotated_image
    # Select the first and only detected pose to visualize.
    pose_landmarks = pose_landmarks_list[0]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
    annotated_image,
    pose_landmarks_proto,
    solutions.pose.POSE_CONNECTIONS,
    solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image
    
def landmarks_to_poses(landmark_results):
    poses = []
    for landmark_result in landmark_results:
        pose = "unknown"
        pose_landmarks_list = landmark_result.pose_landmarks
        if len(pose_landmarks_list) == 0:
            # No pose detected for this person
            poses.append(pose)
            continue
        
        pose_landmarks = pose_landmarks_list[0]
        
        # Extract key landmarks:
        left_hip = pose_landmarks[Landmark.LEFT_HIP]
        right_hip = pose_landmarks[Landmark.RIGHT_HIP]
        left_knee = pose_landmarks[Landmark.LEFT_KNEE]
        right_knee = pose_landmarks[Landmark.RIGHT_KNEE]
        left_ankle = pose_landmarks[Landmark.LEFT_ANKLE]
        right_ankle = pose_landmarks[Landmark.RIGHT_ANKLE]
        left_shoulder = pose_landmarks[Landmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks[Landmark.RIGHT_SHOULDER]
        left_wrist = pose_landmarks[Landmark.LEFT_WRIST]
        right_wrist = pose_landmarks[Landmark.RIGHT_WRIST]
        left_elbow = pose_landmarks[Landmark.LEFT_ELBOW]
        right_elbow = pose_landmarks[Landmark.RIGHT_ELBOW]
        head = pose_landmarks[Landmark.NOSE]  # approximate head

        # --- Standing ---
        standing_landmarks = [
            left_hip, right_hip, 
            left_knee, right_knee, 
            left_shoulder, right_shoulder, 
            left_ankle, right_ankle
            ]
        if is_standing(standing_landmarks):
            pose = "standing"

        # --- Sitting ---
        if pose == "unknown":
            sitting_landmarks = standing_landmarks
            if is_sitting(sitting_landmarks):
                pose = "sitting"
            

        # --- Walking/Stepping ---
        if pose == "unknown":
            walk_landmarks = [
                left_hip, right_hip, 
                left_knee, right_knee, 
                left_ankle, right_ankle
                ]
            if is_walking(walk_landmarks):
                pose = "walking/stepping"

        # --- Arms up (can overlay)
        arms_pose = None
        left_arm_landmarks = [
            left_wrist, 
            left_elbow, 
            left_shoulder, 
            head
            ]
        right_arm_landmarks = [
            right_wrist, 
            right_elbow, 
            right_shoulder, 
            head
            ]
        left_arm_up = is_raised(left_arm_landmarks)
        right_arm_up = is_raised(right_arm_landmarks)
            
        if left_arm_up and right_arm_up:
            arms_pose = "both arms up"
        elif left_arm_up:
            arms_pose = "left arm up"
        elif right_arm_up:
            arms_pose = "right arm up"

        # Combine main pose and arms overlay
        if arms_pose:
            poses.append(f"{pose}, {arms_pose}")
        else:
            poses.append(pose)

    return poses