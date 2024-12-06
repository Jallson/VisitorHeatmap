#!/usr/bin/env python
import os
import cv2
import sys
import getopt
import signal
import time
import numpy as np
from collections import defaultdict
from edge_impulse_linux.image import ImageImpulseRunner

runner = None
# if you don't want to see a camera preview, set this to False
show_camera = True
if sys.platform == 'linux' and not os.environ.get('DISPLAY'):
    show_camera = False

# Heatmap constants
GRID_SIZE = 20
CELL_WIDTH = 640 // GRID_SIZE
CELL_HEIGHT = 640 // GRID_SIZE
DURATION_COLOR_SCALE = [(255, 0, 0), (255, 255, 0), (0, 0, 255)]  # Red, Yellow, Blue
RED_THRESHOLD = 40  # Seconds

heat_map_grid = defaultdict(float)  # Maps (row, col) to time duration

def now():
    return round(time.time() * 1000)

def sigint_handler(sig, frame):
    print('Interrupted')
    if runner:
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

def interpolate_color_and_alpha(duration):
    """Interpolate color and transparency from blue to yellow to red based on duration."""
    # Clamp duration to the maximum threshold
    duration = min(duration, RED_THRESHOLD)

    if duration <= RED_THRESHOLD / 2:
        # Interpolate from blue to yellow
        ratio = duration / (RED_THRESHOLD / 2)
        b = int(DURATION_COLOR_SCALE[2][2] * (1 - ratio))
        g = int(DURATION_COLOR_SCALE[2][1] * ratio + DURATION_COLOR_SCALE[1][1] * (1 - (1 - ratio)))
        r = int(DURATION_COLOR_SCALE[2][0] * (1 - ratio))
    else:
        # Interpolate from yellow to red
        ratio = (duration - RED_THRESHOLD / 2) / (RED_THRESHOLD / 2)
        b = 0  # Blue stays fixed at 0 for yellow-red gradient
        g = int(DURATION_COLOR_SCALE[1][1] * (1 - ratio))
        r = int(DURATION_COLOR_SCALE[1][0] * (1 - ratio) + DURATION_COLOR_SCALE[0][0] * ratio)

    color = (b, g, r)
    # Interpolate alpha (transparency)
    alpha = 0.2 + 0.6 * (duration / RED_THRESHOLD)  # From 20% to 80% transparency
    return color, alpha

def overlay_heatmap(frame, heat_map_grid):
    """Overlay a semi-transparent heat map on the video frame."""
    overlay = frame.copy()
    for (row, col), duration in heat_map_grid.items():
        start_x = col * CELL_WIDTH
        start_y = row * CELL_HEIGHT
        end_x = start_x + CELL_WIDTH
        end_y = start_y + CELL_HEIGHT
        color, alpha = interpolate_color_and_alpha(duration)
        cell_overlay = overlay[start_y:end_y, start_x:end_x].copy()
        cv2.rectangle(cell_overlay, (0, 0), (CELL_WIDTH, CELL_HEIGHT), color, -1)
        overlay[start_y:end_y, start_x:end_x] = cv2.addWeighted(
            cell_overlay, alpha, overlay[start_y:end_y, start_x:end_x], 1 - alpha, 0)
    return overlay

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        print('Error: Invalid arguments')
        sys.exit(2)

    if len(args) == 0:
        print('Usage: python traffic.py <path_to_model.eim> <path_to_video_file>')
        sys.exit(2)

    model = args[0]
    video_path = args[1]  # Path to video file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print('MODEL: ' + modelfile)

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            labels = model_info['model_parameters']['labels']

            # Open the video file instead of camera
            video_capture = cv2.VideoCapture(video_path)
            if not video_capture.isOpened():
                raise Exception(f"Couldn't open video file: {video_path}")

            print(f"Video file {video_path} opened successfully.")
            next_frame = 0

            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    print("End of video file reached or unable to read the frame.")
                    break

                if next_frame > now():
                    time.sleep((next_frame - now()) / 1000)

                # Resize the frame to 640x640 before processing
                resized_frame = cv2.resize(frame, (640, 640))

                # Send the resized frame to the Edge Impulse model for classification
                features, img = runner.get_features_from_image(resized_frame)
                res = runner.classify(features)

                if "bounding_boxes" in res["result"].keys():
                    for bb in res["result"]["bounding_boxes"]:
                        center_x = bb['x'] + bb['width'] // 2
                        center_y = bb['y'] + bb['height'] // 2
                        col = center_x // CELL_WIDTH
                        row = center_y // CELL_HEIGHT
                        heat_map_grid[(row, col)] += 0.1  # Increment detection duration

                # Overlay heatmap
                heatmap_frame = overlay_heatmap(resized_frame, heat_map_grid)

                # Show video feed with heatmap
                if show_camera:
                    cv2.imshow('edgeimpulse with heatmap', heatmap_frame)
                    if cv2.waitKey(1) == ord('q'):
                        break

                next_frame = now() + 100
        finally:
            if runner:
                runner.stop()

if __name__ == "__main__":
    main(sys.argv[1:])
