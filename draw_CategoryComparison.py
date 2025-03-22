import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['Pedestrian', 'Car', 'Bus', 'Bicycle', 'Motorbike']
metrics = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']

# 原始值和变化后的值
data = {
    'Pedestrian': {'Precision': [0.559, 0.588], 'Recall': [0.417, 0.353], 'mAP@0.5': [0.393, 0.394], 'mAP@0.5:0.95': [0.163, 0.175]},
    'Car': {'Precision': [0.617, 0.642], 'Recall': [0.689, 0.691], 'mAP@0.5': [0.700, 0.716], 'mAP@0.5:0.95': [0.422, 0.431]},
    'Bus': {'Precision': [0.597, 0.557], 'Recall': [0.531, 0.490], 'mAP@0.5': [0.584, 0.548], 'mAP@0.5:0.95': [0.362, 0.333]},
    'Bicycle': {'Precision': [0.408, 0.483], 'Recall': [0.083, 0.236], 'mAP@0.5': [0.222, 0.299], 'mAP@0.5:0.95': [0.110, 0.151]},
    'Motorbike': {'Precision': [0.448, 0.531], 'Recall': [0.367, 0.367], 'mAP@0.5': [0.360, 0.390], 'mAP@0.5:0.95': [0.189, 0.203]}
}

# 设置柱状图的位置和宽度
x = np.arange(len(categories))  # 类别的位置
width = 0.15  # 柱子的宽度

# 创建柱状图
fig, ax = plt.subplots(figsize=(12, 8))
for i, metric in enumerate(metrics):
    before = [data[cat][metric][0] for cat in categories]
    after = [data[cat][metric][1] for cat in categories]
    ax.bar(x - width * (len(metrics) / 2 - i), before, width, label=f'{metric} Before')
    ax.bar(x + width * (len(metrics) / 2 - i), after, width, label=f'{metric} After')

# 添加标题和标签
ax.set_xlabel('Categories')
ax.set_ylabel('Values')
ax.set_title('Category-wise Comparison of Evaluation Metrics')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# 保存图片
plt.savefig('evaluation_category_comparison.png', dpi=300)  # 保存为PNG文件，分辨率300dpi
plt.show()