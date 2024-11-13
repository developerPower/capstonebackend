from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # Import CORS
from ultralytics import YOLO
import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import tempfile
import os
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

while True:
    time.sleep(20)  # Adjust the duration as needed

# Load YOLOv8 defect detection model
model_defect = YOLO('...')
track_history = defaultdict(lambda: [])

@app.route('/detect', methods=['POST'])
def detect():
    # Check if a video file is part of the request
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    # Save uploaded video to a temporary location
    video_file = request.files['video']
    temp_video_path = tempfile.mktemp(suffix=".mp4")
    video_file.save(temp_video_path)

    # Open video with OpenCV
    cap = cv2.VideoCapture(temp_video_path)
    assert cap.isOpened(), "Error reading video file"

    # Get video properties
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Prepare output video
    temp_output_path = tempfile.mktemp(suffix=".mp4")
    result = cv2.VideoWriter(temp_output_path,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             fps,
                             (w, h))

    # Process each frame
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Get predictions from the defect detection model
        results_defect = model_defect.track(frame, persist=True, verbose=False)

        # Extract bounding boxes and class information
        boxes = results_defect[0].boxes.xyxy.cpu().numpy()
        clss = results_defect[0].boxes.cls.cpu().numpy()
        track_ids = results_defect[0].boxes.id.int().cpu().numpy() if results_defect[0].boxes.id is not None else None

        # Annotate frame
        annotator = Annotator(frame, line_width=2)

        if track_ids is not None:
            for box, cls, track_id in zip(boxes, clss, track_ids):
                annotator.box_label(box, color=colors(int(cls), True), label=f"Defect {int(cls)}")

                # Store tracking history for each defect
                track = track_history[track_id]
                track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                if len(track) > 30:
                    track.pop(0)

                # Plot tracking history (trail) of defects
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
                cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

        # Write the annotated frame to the result video
        result.write(frame)

    # Release resources
    result.release()
    cap.release()
    os.remove(temp_video_path)  # Clean up the uploaded file

    # Return the annotated video file
    return send_file(temp_output_path, as_attachment=True, download_name="defect_detection.mp4")

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
