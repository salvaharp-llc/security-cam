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
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self.detector.detect(mp_image)

        people = []
        for detection in result.detections:
            people.append(detection.bounding_box)
        return people

    def close(self):
        self.detector.close()