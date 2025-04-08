import numpy as np
import matplotlib.pyplot as plt

# 指标名称
metrics = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]

# 基础模型和改进模型的性能
base_model = [0.526, 0.417, 0.452, 0.249]
modified_model = [0.560, 0.427, 0.469, 0.259]

# 计算提升量
improvement = np.array(modified_model) - np.array(base_model)

# 设置柱状图位置
x = np.arange(len(metrics))
width = 0.35  # 柱子的宽度

# 画图
fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, base_model, width, label="YOLOv10 Model", color='gray', alpha=0.7)
bars2 = ax.bar(x + width/2, modified_model, width, label="YOLOv10+AODNet+CBAM Model", color='blue', alpha=0.7)

# 显示提升数值
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.3f}', ha='center', va='bottom', fontsize=10)

# 标题和标签
ax.set_xlabel("Metric", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Overall Performance Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

plt.ylim(0, 0.6)  # 设置 y 轴范围
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 保存图片
plt.savefig('evaluation_metrics_comparison.png', dpi=300)  # 保存为PNG文件，分辨率300dpi
plt.show()