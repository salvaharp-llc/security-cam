import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from config import OBJECTS_MODEL_PATH, POSE_MODEL_PATH, SCORE_THRESHOLD

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

def draw_landmarks_on_image(image, landmark_results): 
    pose_landmarks_list = landmark_results.pose_landmarks
    annotated_image = np.copy(image)
    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

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
    
def landmarks_to_pose(self, landmarks):
    pass