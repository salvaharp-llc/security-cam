import numpy as np
from mediapipe.framework.formats import landmark_pb2
from config import POSE_THRESHOLDS

def is_standing(standing_landmarks):
    if are_confident(standing_landmarks):
        left_hip, right_hip, left_knee, right_knee, left_shoulder, right_shoulder, left_ankle, right_ankle = standing_landmarks
        shoulder_mid = midpoint(left_shoulder, right_shoulder)
        hip_mid = midpoint(left_hip, right_hip)
        
        legs_straight = leg_is_straight(left_hip, left_knee, left_ankle) and leg_is_straight(right_hip, right_knee, right_ankle)
        torso_upright = vertical_distance(shoulder_mid, hip_mid) > POSE_THRESHOLDS["torso_vertical"] * distance(left_shoulder, right_shoulder)
        hips_above_knees = is_above(left_hip, left_knee) and is_above(right_hip, right_knee)
        open_legs = angle_between(left_knee, midpoint(left_hip, right_hip), right_knee) > POSE_THRESHOLDS["open_legs"]

        if legs_straight and torso_upright and hips_above_knees and not open_legs:
            return True
    return False

def is_sitting(sitting_landmarks):
    if are_confident(sitting_landmarks):
        left_hip, right_hip, left_knee, right_knee, left_shoulder, right_shoulder, left_ankle, right_ankle = sitting_landmarks
        shoulder_mid = midpoint(left_shoulder, right_shoulder)
        hip_mid = midpoint(left_hip, right_hip)
        knee_mid = midpoint(left_knee, right_knee)
        
        legs_bent = leg_is_bent(left_hip, left_knee, left_ankle) and leg_is_bent(right_hip, right_knee, right_ankle)
        torso_upright = is_above(shoulder_mid, hip_mid)
        hip_knee_close = vertical_distance(hip_mid, knee_mid) < POSE_THRESHOLDS["hip_knee_close"] * distance(left_shoulder, right_shoulder)

        if legs_bent and torso_upright and hip_knee_close:
            return True
    return False

def is_walking(walk_landmarks):
    if are_confident(walk_landmarks):
        left_hip, right_hip, left_knee, right_knee, _, _, left_ankle, right_ankle = walk_landmarks
        one_leg_bent = leg_is_bent(left_hip, left_knee, left_ankle) or leg_is_bent(right_hip, right_knee, right_ankle)
        # hip_slope = slope(left_hip, right_hip)
        # hip_tilted = abs(hip_slope) > POSE_THRESHOLDS["hip_tilt"]* distance(left_shoulder, right_shoulder)
        
        if one_leg_bent: # and hip_tilted
            return True
    return False

def is_raised(arm_landmarks):
    if are_confident(arm_landmarks):
        wrist, elbow, shoulder, head = arm_landmarks
        return is_above(wrist, head) and is_above(elbow, shoulder)
    return False

def leg_is_straight(hip, knee, ankle):
    angle_leg = angle_between(hip, knee, ankle)
    return angle_leg > POSE_THRESHOLDS["leg_straight"]

def leg_is_bent(hip, knee, ankle):
    angle_leg = angle_between(hip, knee, ankle)
    return POSE_THRESHOLDS["leg_bent_min"] < angle_leg < POSE_THRESHOLDS["leg_bent_max"]

def angle_between(a, b, c):
    # Build vectors BA and BC
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    
    # Compute cosine using dot product formula
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    # Numerical safety: clip values to [-1, 1] to avoid NaNs due to floating-point errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Convert to degrees
    angle = np.degrees(np.arccos(cos_theta))
    return angle

def is_above(a, b):
    return a.y < b.y

def vertical_distance(a, b):
    return abs(a.y - b.y)

def horizontal_distance(a, b):
    return abs(a.x - b.x)

def midpoint(a, b):
    return landmark_pb2.NormalizedLandmark(
        x=(a.x + b.x) / 2,
        y=(a.y + b.y) / 2,
        z=(a.z + b.z) / 2,
        visibility=(a.visibility + b.visibility) / 2,
        presence=(a.presence + b.presence) / 2,
    )

def slope(a, b, eps=1e-6):
    return (b.y - a.y) / (abs(b.x - a.x) + eps)

def are_confident(landmarks, threshold=POSE_THRESHOLDS["conf_threshold"]):
    for lm in landmarks:
        if lm is None:
            return False
        if lm.presence < threshold or lm.visibility < threshold:
            return False
    return True

def distance(a, b):
    return np.linalg.norm([a.x - b.x, a.y - b.y, a.z - b.z])