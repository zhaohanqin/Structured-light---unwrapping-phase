from generate_graycode_map import GrayCode
import cv2 as cv
import numpy as np
import os

def generate_matched_graycodes():
    """使用GrayCode类生成与解码映射表完全匹配的格雷码图像"""
    print("正在生成匹配的格雷码图像...")
    
    # 创建输出目录
    if not os.path.exists("gray_patterns"):
        os.makedirs("gray_patterns")
    
    # 创建5位格雷码映射对象
    g = GrayCode(5)
    
    # 生成并保存5位格雷码图像
    for i in range(5):
        # 生成格雷码图案，尺寸与现有图像一致(1024x768)
        pattern = g.toPattern(i, cols=1024, rows=768)
        
        # 保存原始格雷码图像
        filename = f"gray_patterns/matched_gray_{i}.png"
        cv.imwrite(filename, pattern)
        print(f"已保存格雷码图像: {filename}")
        
        # 生成二值化格雷码图像
        _, binary = cv.threshold(pattern, 127, 255, cv.THRESH_BINARY)
        bin_filename = f"gray_patterns/matched_binary_{i}.png"
        cv.imwrite(bin_filename, binary)
        print(f"已保存二值化格雷码图像: {bin_filename}")

if __name__ == "__main__":
    generate_matched_graycodes() 