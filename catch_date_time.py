from datetime import datetime, timedelta

def get_time():
    now = datetime.now().time()
    return timedelta(
        hours=now.hour,
        minutes=now.minute,
        seconds=now.second,
    )

def get_date():
    return datetime.now().strftime("%Y-%m-%d")

def frame_to_time(frame_number, fps):
    seconds = frame_number / fps
    return timedelta(seconds=round(seconds))