from ultralytics import YOLOv10
import os

# load model
model = YOLOv10("yolov10n-test.yaml")  # model structure
# load pre-trained weights
model = YOLOv10("weights/yolov10n.pt")

# single thread, remove if workers is not 0
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# train model
if __name__ == '__main__':
    model.train(data="data.yaml", imgsz=640, batch=16, epochs=100, workers=0)