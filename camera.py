import cv2
import os
from config import DELAY, TEST_VIDEO_PATH
from catch_date_time import get_time, get_date, frame_to_time
from detection import PeopleDetector, PoseDetector, draw_landmarks_on_image, landmarks_to_poses
from create_log import EventLogger

def get_camera(debug, verbose, camera_port = 0):
    try:
        camera_port = int(camera_port)
    except ValueError:
        print(f"Invalid camera port: {camera_port}. Must be an integer.")
        return
    cap = cv2.VideoCapture(camera_port)

    people_detector = PeopleDetector()
    pose_detector = PoseDetector()

    log = EventLogger(get_date())

    while True:
        success, image = cap.read()
        if not success:
            print('Error: unable to read frame')
            break
        
        time = get_time()
        people, landmark_results = process_frame(image, people_detector, pose_detector)
        poses = landmarks_to_poses(landmark_results)
        if people:
            if not log.current_event:
                log.start_event(time)
            log.add_frame(time, len(people), poses)
            end_time = time
        elif log.current_event:
            log.end_event(end_time)

        if verbose:   
            print(f'{len(people)} detected at {time}')
            for pose in poses:
                print(pose)

        if debug:
            plot_people(time, people, landmark_results)

    if log.current_event:
        log.end_event(end_time)

    cap.release()
    people_detector.close()
    pose_detector.close()

def get_video(debug, verbose, video_path = TEST_VIDEO_PATH):
    if not os.path.isfile(video_path):
        print(f"Video file not found: {video_path}")
        return
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_step = int(DELAY * fps)
    frame_number = 0

    people_detector = PeopleDetector()
    pose_detector = PoseDetector()

    log = EventLogger(os.path.basename(video_path))

    while frame_number < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, image = cap.read()
        if not success:
            print('Error: unable to read frame')
            break
        
        time = frame_to_time(frame_number, fps)
        people, landmark_results = process_frame(image, people_detector, pose_detector)
        poses = landmarks_to_poses(landmark_results)
        if people:
            if not log.current_event:
                log.start_event(time)
            log.add_frame(time, len(people), poses)
            end_time = time
        elif log.current_event:
            log.end_event(end_time)

        if verbose:   
            print(f'{len(people)} detected at {time}')
            for pose in poses:
                print(pose)

        if debug:
            plot_people(time, people, landmark_results)

        frame_number += frame_step

    if log.current_event:
        log.end_event(end_time)

    cap.release()
    people_detector.close()
    pose_detector.close()

def process_frame(image, people_detector, pose_detector):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    people_boxes = people_detector.detect(img_rgb)
    landmark_results = []
    people_images = []
    for bbox in people_boxes:
        x_min = int(bbox.origin_x)
        y_min = int(bbox.origin_y)
        x_max = int(bbox.origin_x + bbox.width)
        y_max = int(bbox.origin_y + bbox.height)

        person_img = img_rgb[y_min:y_max, x_min:x_max].copy()
        people_images.append(person_img)

        pose_result = pose_detector.detect(person_img)
        landmark_results.append(pose_result)

    return people_images, landmark_results

def plot_people(time, people_images, landmark_results):
    for idx in range(len(people_images)):
        annotated_image = draw_landmarks_on_image(people_images[idx], landmark_results[idx])
        cv2.imshow(f'Person {idx+1} detected at {time}', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    
    cv2.waitKey(0)
    
    for idx in range(len(people_images)):
        cv2.destroyWindow(f'Person {idx+1} detected at {time}')
