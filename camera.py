import cv2
import os
from config import DELAY, TEST_VIDEO_PATH
from catch_date_time import get_time, frame_to_time
from detection import PeopleDetector

def get_camera(camera_port = 0):
    try:
        camera_port = int(camera_port)
    except ValueError:
        print(f"Invalid camera port: {camera_port}. Must be an integer.")
        return
    cap = cv2.VideoCapture(camera_port)

    while True:
        success, img = cap.read()
        if not success:
            print('Error: unable to read frame')
            break
        
        time = get_time()
        cv2.imshow(time, img)

        if cv2.waitKey(int(DELAY * 1000)) != -1:
            print('Stoping the program')
            break

    cap.release()

def get_video(video_path = TEST_VIDEO_PATH):
    if not os.path.isfile(video_path):
        print(f"Video file not found: {video_path}")
        return
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_step = DELAY * fps
    frame_number = 0

    people_detector = PeopleDetector()

    while frame_number <= total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, img = cap.read()
        if not success:
            print('Error: unable to read frame')
            break
        
        time = frame_to_time(frame_number, fps)
        people = people_detector.detect(img)
        print(f'{len(people)} detected at {time}')

        frame_number += frame_step

    cap.release()
    people_detector.close()