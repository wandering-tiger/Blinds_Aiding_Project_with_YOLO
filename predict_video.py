import os
import cv2
import supervision as sv
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLOv10
from tqdm import tqdm
from ViewTransformer import ViewTransformer

SOURCE_VIDEO_PATH = r"datasets/video1.mp4"
TARGET_VIDEO_PATH = r"video1-result.mp4"
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
MODEL_RESOLUTION = 1280

SOURCE = np.array([
    [662, 753],
    [1109, 737],
    [1588, 934],
    [362, 1010]
])

TARGET = np.array([
    [0, 0],
    [5.3, 0],
    [5.3, 9.7],
    [0, 9.7],
])

# SOURCE = np.array([
#     [1252, 787],
#     [2298, 803],
#     [5039, 2159],
#     [-550, 2159]
# ])
#
# TARGET = np.array([
#     [0, 0],
#     [24, 0],
#     [24, 249],
#     [0, 249],
# ])

RANGE = np.array([
    [1252, 787],
    [2298, 803],
    [5039, 2159],
    [-550, 2159]
])

view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

# 加载预训练模型
model = YOLOv10(r"runs/detect/train15/weights/best.pt")

# # 获取预测目录下视频的路径
# video_path = r"datasets/video1.mp4"
# cap = cv2.VideoCapture(video_path)


# output_folder = r"temp_videos"
# results = model.track(source=video_path, save_dir=output_folder, save=True)

video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)

# tracer initiation
byte_track = sv.ByteTrack(
    frame_rate=video_info.fps
)

# annotators configuration
thickness = sv.calculate_optimal_line_thickness(
    resolution_wh=video_info.resolution_wh
)
text_scale = sv.calculate_optimal_text_scale(
    resolution_wh=video_info.resolution_wh
)
bounding_box_annotator = sv.BoxAnnotator(
    thickness=thickness
)
label_annotator = sv.LabelAnnotator(
    text_scale=text_scale,
    text_thickness=thickness,
    text_position=sv.Position.BOTTOM_CENTER
)
trace_annotator = sv.TraceAnnotator(
    thickness=thickness,
    trace_length=video_info.fps * 2,
    position=sv.Position.BOTTOM_CENTER
)

polygon_zone = sv.PolygonZone(
    polygon=RANGE
)

coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

# open target video
with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:

    # loop over source video frame
    for frame in tqdm(frame_generator, total=video_info.total_frames):

        result = model(frame, imgsz=MODEL_RESOLUTION, verbose=False)[0]
        # print("Model result:", result)
        detections = sv.Detections.from_ultralytics(result)
        # print("Detections:", detections)

        # filter out detections by class and confidence
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        detections = detections[detections.class_id != 0]

        # filter out detections outside the zone
        detections = detections[polygon_zone.trigger(detections)]

        # refine detections using non-max suppression
        detections = detections.with_nms(IOU_THRESHOLD)

        # pass detection through the tracker
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        )

        # calculate the detections position inside the target RoI
        points = view_transformer.transform_points(points=points).astype(int)

        # store detections position
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)

        # format labels
        labels = []

        for tracker_id in detections.tracker_id:
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                # calculate speed
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time
                labels.append(f"#{tracker_id} {int(speed)} m/s")

        # annotate frame
        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        # add frame to target video
        sink.write_frame(annotated_frame)
