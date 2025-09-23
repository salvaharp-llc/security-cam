import argparse
from camera import get_camera, get_video
def main():
    parser = argparse.ArgumentParser(
        description='Security Cam: Run with camera or video file.'
    )
    parser.add_argument(
        'mode',
        choices=['camera', 'video'],
        help='Use recorded video or live camera'
    )
    parser.add_argument(
        'source',
        nargs='?',
        default=None,
        help='Optional: path to video file or camera port (default: 0 for camera, config for video)'
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    args = parser.parse_args()

    if args.mode == 'video':
        if args.source:
            get_video(args.debug, args.source)
        else:
            get_video(args.debug)
    elif args.mode == 'camera':
        if args.source:
            get_camera(args.source)
        else:
            get_camera()

if __name__ == "__main__":
    main()
