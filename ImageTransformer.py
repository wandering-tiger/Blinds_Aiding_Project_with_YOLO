import cv2
import numpy as np
from ViewTransformer import ViewTransformer

image = cv2.imread("road_origin.jpg")
SOURCE = np.array([
    [665, 751],
    [1110, 737],
    [1160, 765],
    [634, 781]
])

TARGET = np.array([
    [0, 0],
    [5.3, 0],
    [5.3, 2],
    [0, 2],
])

view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

# height, width, channels = image.shape
# output_width = width
# output_height = height
# transformed_image = view_transformer.transform_image(image, output_width, output_height)
# cv2.imwrite("transformed.jpg", transformed_image)

test_points = np.array([
    [500, 912],
    [1322, 912]
])

# 执行透视变换
transformed_points = view_transformer.transform_points(test_points)

# 输出结果
print("原始坐标:")
print(test_points)
print("\n转换后的坐标:")
print(transformed_points)