import numpy as np
import cv2
import os

# 参数设置
width = 1024       # 图像宽度
height = 768       # 图像高度
frequency = 10     # 条纹频率（周期数）
I0 = 127           # 亮度基值（中值）
A = 127            # 亮度幅度（最大幅度）

# 保存目录
output_dir = "fringe_patterns"
os.makedirs(output_dir, exist_ok=True)

# 相移列表（四步相移）
phase_shifts = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

# 生成图像
for idx, phi in enumerate(phase_shifts):
    # 创建空图像
    pattern = np.zeros((height, width), dtype=np.uint8)

    # 生成一维条纹并广播到二维
    x = np.arange(width)
    fringe = I0 + A * np.cos(2 * np.pi * frequency * x / width + phi)
    fringe = np.tile(fringe, (height, 1))  # 拓展为整幅图像
    fringe = np.clip(fringe, 0, 255).astype(np.uint8)

    # 保存图像
    filename = f"{output_dir}/phase_shift_{idx}.png"
    cv2.imwrite(filename, fringe)
    print(f"Saved: {filename}")
