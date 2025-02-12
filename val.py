import os
from ultralytics import YOLOv10

# 加载预训练模型
model = YOLOv10(r"runs/detect/train15/weights/best.pt")

# 获取预测目录下所有图片的路径
predict_dir = r"datasets/dataset1/images/val"
image_paths = [os.path.join(predict_dir, img) for img in os.listdir(predict_dir) if
               img.endswith(('.jpg', '.jpeg', '.png'))]


output_folder = r"temp_photos"
# 对每张图片进行推理
for image_path in image_paths:
    results = model.predict(image_path)
    # 生成输出文件路径
    image_name = os.path.basename(image_path)  # 获取图片名称
    save_path = os.path.join(output_folder, f"pred_{image_name}")  # 生成保存路径

    # 保存预测结果图片
    results[0].save(save_path)