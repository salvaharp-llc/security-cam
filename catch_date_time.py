from datetime import datetime, timedelta

def get_time():
    return datetime.now() # .strftime("%H:%M:%S")

def get_date():
    return datetime.now() # .strftime("%Y-%m-%d")

def frame_to_time(frame_number, fps):
    seconds = frame_number / fps
    return timedelta(seconds=round(seconds))