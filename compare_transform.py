import cv2
import supervision as sv
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from ultralytics import YOLOv10
from tqdm import tqdm
from ViewTransformer import ViewTransformer

# **Video paths**
SOURCE_VIDEO_PATH = r"datasets/video1.mp4"
CSV_OUTPUT_PATH = r"speed_comparison.csv"

# **YOLO Detection Settings**
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
MODEL_RESOLUTION = 1280
SPEED_THRESHOLD = 2  # Minimum speed threshold for valid data

# **Initialize video capture to get frame rate**
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
cap.release()

sample_interval = fps // 4  # Sample frequency

# **Transformation matrices**
SOURCE_1 = np.array([
    [1252, 787],
    [2298, 803],
    [5039, 2159],
    [-550, 2159]
])
TARGET_1 = np.array([
    [0, 0],
    [24, 0],
    [24, 249],
    [0, 249],
])

# **Second transformation matrix (scaled dynamically)**
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

sample_source_width = 1702
sample_source_height = 1276
ratio_width = width / sample_source_width
ratio_height = height / sample_source_height

SOURCE_2 = np.array([
    [662 * ratio_width, 753 * ratio_height],
    [1109 * ratio_width, 737 * ratio_height],
    [1588 * ratio_width, 934 * ratio_height],
    [362 * ratio_width, 1010 * ratio_height]
])
TARGET_2 = np.array([
    [0, 0],
    [5.3, 0],
    [5.3, 9.7],
    [0, 9.7],
])

# **Initialize View Transformers**
view_transformer_1 = ViewTransformer(source=SOURCE_1, target=TARGET_1)
view_transformer_2 = ViewTransformer(source=SOURCE_2, target=TARGET_2)

# **Load YOLOv10 model**
model = YOLOv10(r"runs/detect/train15/weights/best.pt")

# **Get video information**
video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)

# **Tracker Initialization**
byte_track = sv.ByteTrack(frame_rate=video_info.fps)

# **Store past positions to compute speed**
coordinates_1 = defaultdict(lambda: deque(maxlen=video_info.fps))
coordinates_2 = defaultdict(lambda: deque(maxlen=video_info.fps))

CLASS_NAMES = model.names
speed_data = []  # Store results

# **Process Video Frames**
for frame_idx, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames)):
    # **Process only one frame every half second**
    if frame_idx % sample_interval != 0:
        continue

    result = model(frame, imgsz=MODEL_RESOLUTION, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)

    # **Filter detections**
    detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
    detections = detections[detections.class_id != 0]
    detections = detections.with_nms(IOU_THRESHOLD)

    detections = byte_track.update_with_detections(detections=detections)

    points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

    if len(points) == 0:
        continue

    # **Apply both transformations**
    points_1 = view_transformer_1.transform_points(points=points).astype(int)
    points_2 = view_transformer_2.transform_points(points=points).astype(int)

    # **Store positions for speed calculation**
    for tracker_id, (y1, y2) in zip(detections.tracker_id, zip(points_1[:, 1], points_2[:, 1])):
        coordinates_1[tracker_id].append((y1, frame_idx))
        coordinates_2[tracker_id].append((y2, frame_idx))

    # **Compute speeds and compare**
    for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
        class_name = CLASS_NAMES.get(class_id, "Unknown")

        if len(coordinates_1[tracker_id]) < 2 or len(coordinates_2[tracker_id]) < 2:
            continue  # Skip if not enough data points

        # **Speed Calculation (Perspective 1)**
        y1_start, frame_start = coordinates_1[tracker_id][0]
        y1_end, frame_end = coordinates_1[tracker_id][-1]
        delta_t = (frame_end - frame_start) / fps
        speed_1 = abs(y1_end - y1_start) / delta_t if delta_t > 0 else 0

        # **Speed Calculation (Perspective 2)**
        y2_start, _ = coordinates_2[tracker_id][0]
        y2_end, _ = coordinates_2[tracker_id][-1]
        speed_2 = abs(y2_end - y2_start) / delta_t if delta_t > 0 else 0

        # **Filter out unrealistic speeds**
        if speed_1 < SPEED_THRESHOLD or speed_2 < SPEED_THRESHOLD:
            continue

        # **Compute Speed Difference Percentage**
        speed_diff = abs(speed_1 - speed_2) / speed_1 * 100 if speed_1 > 0 else 0

        # **Store results**
        speed_data.append([frame_idx, class_name, tracker_id, speed_1, speed_2, speed_diff])

# **Save speed comparison results**
speed_df = pd.DataFrame(speed_data, columns=["Frame", "Class", "Tracker_ID", "Speed_1 (m/s)", "Speed_2 (m/s)", "Speed_Diff (%)"])
speed_df.to_csv(CSV_OUTPUT_PATH, index=False)
print(f"Speed comparison saved to {CSV_OUTPUT_PATH}")
