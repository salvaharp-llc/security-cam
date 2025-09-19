import argparse
from camera import get_camera, get_video
def main():
    parser = argparse.ArgumentParser(
        description='Security Cam: Run with camera or video file.'
    )
    parser.add_argument(
        'mode',
        choices=['camera'],
        help='Mode to run: camera'
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Use recorded video instead of live camera'
    )
    args = parser.parse_args()

    if args.test:
        get_video()
    elif args.mode == 'camera':
        get_camera()

if __name__ == "__main__":
    main()
