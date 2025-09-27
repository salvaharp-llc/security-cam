DELAY = 2 # Seconds between each reading

TEST_VIDEO_PATH = "recordings/video-for-pose-detection-1-1080x1920-30fps.mp4"
OBJECTS_MODEL_PATH = 'models/efficientdet_lite0.tflite'
POSE_MODEL_PATH = 'models/pose_landmarker_full.task'

LOGS_PATH = "logs/"

SCORE_THRESHOLD = 0.35
POSE_THRESHOLDS = {
    "leg_straight": 160,
    "leg_bent_min": 60,
    "leg_bent_max": 155,
    "torso_vertical": 0.1,
    "hip_knee_close": 0.3,
    "conf_threshold": 0.4,
}