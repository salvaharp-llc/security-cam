import mediapipe as mp
import cv2
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
        result = self.landmarker.detect(mp_image)
        return result
    
    def landmark_to_pose(self, landmark):
        pass

    def close(self):
        self.landmarker.close()