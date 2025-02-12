import os

# 定义数据集路径
dataset_path = r'/mnt/shared/home/b30ubckb1/Project_YOLO/datasets/dataset1/images'

# 定义保存路径
output_paths = {
    'train': r'/mnt/shared/home/b30ubckb1/Project_YOLO/datasets/dataset1/train.txt',
    'val': r'/mnt/shared/home/b30ubckb1/Project_YOLO/datasets/dataset1/val.txt',
}

# 文件夹映射到相应的输出路径
for split, output_path in output_paths.items():
    # 获取当前数据集（train/val/test）文件夹中的所有图片路径
    folder_path = os.path.join(dataset_path, split)
    image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.jpg', '.png'))]

    # 将图片路径写入到对应的txt文件
    with open(output_path, 'w') as f:
        for image_path in image_paths:
            f.write(image_path + '\n')

    print(f"{split}.txt 文件已生成，包含 {len(image_paths)} 张图片路径。")
