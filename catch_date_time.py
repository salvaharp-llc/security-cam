from datetime import datetime, timedelta

def get_time():
    date = datetime.now().isoformat(sep=' ', timespec='seconds')
    _, time = date.split(' ', 1)
    return time

def frame_to_time(frame_number, fps):
    seconds = int(frame_number / fps)
    return str(timedelta(seconds=seconds))