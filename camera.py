import cv2
from config import DELAY, TEST_VIDEO_PATH
from catch_date_time import get_time, frame_to_time
from detection import PeopleDetector

def get_camera():
    cap = cv2.VideoCapture(0)
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

def get_video():
    cap = cv2.VideoCapture(TEST_VIDEO_PATH)
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