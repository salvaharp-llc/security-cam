# Security Cam

## Overview

**Security Cam** is an AI-powered video analysis tool for detecting and analyzing people and their poses in video streams or camera feeds.  
It uses machine learning models (via MediaPipe) to:

- Detect people in each frame from a live camera or video file.
- Crop each detected person and analyze their pose (standing, sitting, walking, arms raised, etc.).
- Log detection events and pose statistics to JSON files for later review.
- Optionally display annotated images of detected people and their poses in real time.

## Features

- Works with both live camera feeds and prerecorded video files.
- Pose classification for each detected person.
- Configurable detection thresholds and delay between frames.
- Command-line interface for flexible operation and debugging.
- Event logging for security and analytics.

## Usage

Run from the command line:

```bash
python main.py camera            # Use live camera (default port 0)
python main.py camera 1         # Use camera at port 1
python main.py video            # Use default test video
python main.py video path/to/video.mp4  # Use custom video file

# Optional flags:
-h / --help       # Print usage info
-d / --debug      # Show annotated images of detected people and poses
-v / --verbose    # Print detection and pose info in the terminal
```

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe

All Python dependencies are listed in `pyproject.toml`.  
To install them, run:

```bash
pip install .
```

See `config.py` for model paths and detection thresholds.

## Output

- Annotated images (if debug mode is enabled)
- JSON logs of detection events and pose statistics in the `logs/` directory

## Future Improvements

- Advanced pose interpretation from landmarks using a custom machine learning model.
- Analytics dashboard for event statistics, such as average number of people detected and most common poses.
- Improved real-time performance and support for additional camera types.
- Enhanced logging and alerting features.
