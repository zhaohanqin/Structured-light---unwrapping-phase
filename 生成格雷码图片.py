import numpy as np
import cv2
import os

# 设置图像参数
width = 1024   # 图像宽度（投影仪的宽度）
height = 768   # 图像高度
n_bits = 5     # Gray 编码位数（将生成 5 张图）

# 保存路径
output_dir = "gray_patterns"
os.makedirs(output_dir, exist_ok=True)

# 遍历每一列，计算其 gray 码
for bit in range(n_bits):
    gray_img = np.zeros((height, width), dtype=np.uint8)
    
    for x in range(width):
        binary = x
        gray = binary ^ (binary >> 1)          # 二进制转 gray 码
        bit_val = (gray >> (n_bits - bit - 1)) & 1  # 提取对应位的 bit
        
        gray_img[:, x] = 255 * bit_val          # 显示为黑白条纹
    
    filename = f"{output_dir}/gray_bit_{bit}.png"
    cv2.imwrite(filename, gray_img)
    print(f"Saved: {filename}")
