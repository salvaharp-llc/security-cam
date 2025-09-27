import os
import json
import copy
from datetime import timedelta
from config import LOGS_PATH

class EventLogger:
    def __init__(self, name, clear=True):
        self.name = name
        self.file_path = f"{LOGS_PATH}{name}.json"
        self.current_event = None
        if clear:
            delete_log(self.file_path)

    def start_event(self, start_time):
        self.current_event = {
            "start": start_time,
            "frames": []
        }

    def add_frame(self, frame_time, num_people, poses):
        if self.current_event:
            self.current_event["frames"].append({
                "time": frame_time,
                "num_people": num_people,
                "poses": poses
            })

    def end_event(self, end_time):
        if self.current_event:
            self.current_event["end"] = end_time
            self.current_event["duration"] = (
                end_time - self.current_event["start"]
            )
            if self.current_event["duration"].total_seconds() < 0:
                self.current_event["duration"] += timedelta(days=1)
            summarized = summarize_event(self.current_event)
            save_log(self.file_path, summarized)
            self.current_event = None


def summarize_event(event):
    summarized = copy.deepcopy(event)
    for idx, frame in enumerate(event["frames"]):
        summarized_poses = {}
        for pose in frame["poses"]:
            if pose not in summarized_poses:
                summarized_poses[pose] = 0
            summarized_poses[pose] += 1
        summarized["frames"][idx]["poses"] = summarized_poses
    return summarized

def save_log(file_path, event):
    if os.path.isfile(file_path):
        with open(file_path, "r") as f:
            log = json.load(f)
    else:
        log = {"events": []}
    
    log["events"].append(event)

    with open(file_path, "w") as f:
        json.dump(log, f, indent=4, default=str)

def delete_log(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)