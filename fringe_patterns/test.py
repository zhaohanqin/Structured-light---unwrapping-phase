import numpy as np
import cv2
import os

# 图像尺寸
width = 1024
height = 768

# 正弦条纹参数
I0 = 127.5         # 平均亮度
Im = 127.5         # 亮度振幅（范围 0~127.5）
f = 10 / height    # 条纹频率（每图像有10条波）

# 输出文件夹
output_dir = "fringe_patterns"
os.makedirs(output_dir, exist_ok=True)

# 四步相移的相位值
phase_shifts = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

for i, phi in enumerate(phase_shifts):
    img = np.zeros((height, width), dtype=np.uint8)

    # 垂直方向生成条纹，y 变化，x 每列相同
    for y in range(height):
        intensity = I0 + Im * np.cos(2 * np.pi * f * y + phi)
        img[y, :] = np.clip(intensity, 0, 255)

    filename = os.path.join(output_dir, f"vertical_phase_{i}.png")
    cv2.imwrite(filename, img)
    print(f"Saved: {filename}")
