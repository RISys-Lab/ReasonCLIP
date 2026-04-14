import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# 1️⃣ 数据集标签（已固定）
# ==================================================
labels = [
    # 第一组
    "WinoGAViL", "RCLIP-V1", "RCLIP-V2", "RCLIP-V3",
    # 第二组
    "WhatsUp", "Valse", "Crepe", "SugarCREPE",
    # 第三组
    "COCO", "Flickr", "Urban1k", "RCLIP"
]

num_vars = len(labels)

# ==================================================
# 2️⃣ 在这里填写你的真实数据
# ==================================================
raw_data_clip = {
    "CLIP": [
        50.2, 22.9, 17.7, 28.8,
        41.8, 68.8, 20.9, 74.8,
        47.5, 41.6, 65.0, 44.6,
    ],
    "S1": [
        54.8, 27.2, 19.3, 33.3,
        42.7, 71.2, 16.1, 74.4,
        56.0, 51.5, 82.5, 52.9
    ],
    "S2": [
        60.2, 29.5, 18.0, 30.3,
        46.3, 76.3, 24.8, 78.1,
        54.2, 49.0, 77.4, 55.5
    ],
    "S0-Des.": [
        54.8, 26.0, 18.2, 31.4,
        47.3, 76.3, 20.1, 79.5,
        55.7, 49.5, 81.8, 42.6
    ],
    "S0-Rea.": [
        58.0, 28.8, 19.2, 32.1,
        41.5, 71.2, 17.3, 73.7,
        55.0, 48.7, 81.9, 46.5
    ],
}

raw_data_siglip = {
    "SigLIP": [
        60.8, 24.4, 22.2, 26.3,
        47.6, 72.0, 18.0, 83.0,
        63.5, 60.6, 74.0, 57.0
    ],
    "S1": [
        64.0, 30.8, 25.8, 29.4,
        48.2, 77.5, 23.7, 86.9,
        65.1, 60.8, 79.9, 56.9
    ],
    "S2": [
        63.5, 30.7, 24.1, 28.4,
        49.7, 76.3, 20.1, 84.1,
        65.5, 63.6, 79.6, 61.0
    ],
    "S0-Des.": [
        62.1, 31.5, 22.2, 27.9,
        50.8, 76.8, 21.7, 87.5,
        64.9, 60.8, 81.3, 56.5
    ],
    "S0-Rea.": [
        64.9, 25.8, 24.4, 29.4,
        47.1, 71.8, 12.5, 73.6,
        59.4, 59.3, 80.2, 60.2
    ],
}

# 👇 【关键修复】：把数据和对应的保存名字直接用 zip 绑起来遍历
for raw_data, save_name in zip([raw_data_clip, raw_data_siglip], ["clip", "siglip"]):
    # 3️⃣ 基本检查（防止填错）
    for model, values in raw_data.items():
        if len(values) != num_vars:
            raise ValueError(f"{model} 数据数量不等于 {num_vars}")

    # ==================================================
    # 4️⃣ 带视觉缓冲的独立区间缩放 (Soft Min-Max Scaling)
    # ==================================================
    data_matrix = np.array(list(raw_data.values()))

    col_min = np.min(data_matrix, axis=0)
    col_max = np.max(data_matrix, axis=0)

    base = 20
    top = 100

    visual_min = 40
    visual_max = 95

    scaled_matrix = np.zeros_like(data_matrix, dtype=float)

    for i in range(num_vars):
        if col_max[i] == col_min[i]:
            scaled_matrix[:, i] = base
        else:
            scaled_matrix[:, i] = visual_min + (data_matrix[:, i] - col_min[i]) / (col_max[i] - col_min[i]) * (
                    visual_max - visual_min)

    model_data = dict(zip(raw_data.keys(), scaled_matrix))

    # ==================================================
    # 5️⃣ 雷达图角度设置
    # ==================================================
    angle_offset = np.pi / num_vars
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False) + angle_offset
    angles = np.concatenate((angles, [angles[0]]))

    # ==================================================
    # 6️⃣ 画图
    # ==================================================
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)

    ax.grid(True, color="gray", alpha=0.3)
    ax.spines['polar'].set_visible(False)

    # ==================================================
    # 🎨 轴标签与三组灰度着色逻辑
    # ==================================================
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=14, fontweight='bold')

    # 用 pad 统一把所有标签往外推
    ax.tick_params(axis='x', pad=10)

    # 设定三组同等权重的深色（暗藏青 -> 暗砖红 -> 暗墨绿）
    group_colors = ["#1F4E79"] * 4 + ["#803B32"] * 4 + ["#2E6642"] * 4

    for i, (label, angle) in enumerate(zip(ax.get_xticklabels(), angles[:-1])):
        label.set_horizontalalignment('center')
        label.set_verticalalignment('center')
        label.set_color(group_colors[i])

    ax.set_yticklabels([])

    colors = [
        "#6C9A8B",  # 更柔和的绿
        "#1f77b4",  # 蓝色
        "#9467bd",  # 紫色
        "#b0b0b0",  # 灰色1
        "#a9746e",  # 砖红灰
    ]

    for (model, values), color in zip(model_data.items(), colors):
        values = np.concatenate((values, [values[0]]))
        ax.fill(angles, values, color=color, alpha=0.13)
        ax.plot(angles, values, linewidth=2, color=color, label=model)

    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()

    # 👇 【关键修复】：直接使用当前循环拿到的名字保存
    plt.savefig(save_name + ".svg", format="svg", bbox_inches="tight")
    plt.show()