import matplotlib.pyplot as plt
import numpy as np

# 数据
metrics = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']
yolov10n_dehaze_cbam1 = [0.560, 0.427, 0.469, 0.259]
yolov10n_test = [0.526, 0.417, 0.452, 0.249]
changes = [0.034, 0.010, 0.017, 0.010]

# 设置柱状图的位置和宽度
x = np.arange(len(metrics))  # 每个指标的位置
width = 0.35  # 柱子的宽度

# 创建柱状图
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, yolov10n_dehaze_cbam1, width, label='YOLOv10n-modified')
rects2 = ax.bar(x + width/2, yolov10n_test, width, label='YOLOv10n-base')

# 添加文本标签
for i in range(len(metrics)):
    ax.text(x[i] - width/2, yolov10n_dehaze_cbam1[i] + 0.01, f'{yolov10n_dehaze_cbam1[i]:.3f}', ha='center', va='bottom')
    ax.text(x[i] + width/2, yolov10n_test[i] + 0.01, f'{yolov10n_test[i]:.3f}\n↑ {changes[i]:.3f}', ha='center', va='bottom')

# 添加标题和标签
ax.set_xlabel('Metrics')
ax.set_ylabel('Values')
ax.set_title('Comparison of Evaluation Metrics')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# 保存图片
plt.savefig('evaluation_metrics_comparison.png', dpi=300)  # 保存为PNG文件，分辨率300dpi
plt.show()