import cv2
from config import DELAY
from catch_date_time import get_time

def get_camera():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            print('Error: unable to read frame')
            break
        
        time = get_time()
        cv2.imshow(time, img)

    cap.release()

def get_video():
    cap = cv2.VideoCapture("recordings/dummy.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_step = DELAY * fps
    frame_number = 0

    while frame_number <= total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, img = cap.read()
        if not success:
            print('Error: unable to read frame')
            break
        
        time = get_time()
        cv2.imshow(time, img)

        if cv2.waitKey(int(DELAY * 1000)) != -1:
            print('Stoping the program')
            break

        frame_number += frame_step

    cap.release()