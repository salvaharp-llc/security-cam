from datetime import datetime

def get_time():
    date = datetime.now().isoformat(sep=' ', timespec='seconds')
    _, time = date.split('-', 1)
    return time