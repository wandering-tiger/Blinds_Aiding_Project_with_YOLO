import numpy as np
import matplotlib.pyplot as plt

# 定义类别
categories = ["Pedestrian", "Car", "Bus", "Bicycle", "Motorbike"]

# 定义指标的基准模型和修改后模型数值
precision_base = [0.559, 0.617, 0.597, 0.408, 0.448]
precision_modified = [0.588, 0.642, 0.557, 0.483, 0.531]

recall_base = [0.417, 0.689, 0.531, 0.083, 0.367]
recall_modified = [0.353, 0.691, 0.490, 0.236, 0.367]

map50_base = [0.393, 0.700, 0.584, 0.222, 0.360]
map50_modified = [0.394, 0.716, 0.548, 0.299, 0.390]

map5095_base = [0.163, 0.422, 0.362, 0.110, 0.189]
map5095_modified = [0.175, 0.431, 0.333, 0.151, 0.203]

# 颜色方案
base_color = "gray"  # Base Model 灰色
modified_colors = ["red", "blue", "green", "orange", "purple"]  # Modified Model 类别颜色

# 画图
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
x = np.arange(len(categories))  # X轴位置
width = 0.35  # 柱状图宽度

# 设置子图
metrics = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]
base_data = [precision_base, recall_base, map50_base, map5095_base]
modified_data = [precision_modified, recall_modified, map50_modified, map5095_modified]

for i, ax in enumerate(axes.flat):
    for j, category in enumerate(categories):
        ax.bar(x[j] - width/2, base_data[i][j], width, color=base_color, alpha=0.7, label="Base Model" if j == 0 else None)
        ax.bar(x[j] + width/2, modified_data[i][j], width, color=modified_colors[j], alpha=0.9, label=category if i == 0 else None)

    ax.set_title(metrics[i], fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Score")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

# **在外部手动创建图例**
handles = [plt.Rectangle((0,0),1,1, color=base_color, alpha=0.7, label="Base Model")] + \
          [plt.Rectangle((0,0),1,1, color=modified_colors[i], alpha=0.9, label=categories[i]) for i in range(len(categories))]

fig.legend(handles, ["Base Model"] + categories, loc="lower center", ncol=6, fontsize=10)

# 调整布局
plt.suptitle("Performance Comparison Across Categories", fontsize=14, fontweight="bold", color="black")
plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 预留下方空间

# 保存图片
plt.savefig('evaluation_category_comparison.png', dpi=300)  # 保存为PNG文件，分辨率300dpi
plt.show()