import numpy as np
import cv2 as cv
import os
import math
import argparse

class GrayCode():
    """
    格雷码生成与解码类
    
    该类实现了格雷码的生成、编码和解码功能，用于辅助相位解包裹。
    格雷码是一种二进制编码方式，相邻编码之间仅有一位不同，
    这一特性使其在相位解包裹中特别有用。
    """
    
    codes = np.array([])  # 格雷码矩阵
    code2k = {}  # 格雷码到k值的映射
    k2v = {}     # k值到v值的映射
    v2k = {}     # v值到k值的映射
    
    def __init__(self, n: int = 5):
        """
        初始化格雷码生成器
        
        参数:
            n: 格雷码位数，默认为5位
        """
        self.n = n
        self.codes = self.__formCodes(self.n)
        # 从格雷码转换到k
        for k in range(2 ** n):
            self.code2k[self.__code2k(k)] = k
        # 从格雷码转换到v
        for k in range(2 ** n):
            self.k2v[k] = self.__k2v(k)
        # 从v转换到k（idx）
        for k, v in self.k2v.items():
            self.v2k[v] = k

    @staticmethod
    def __createGrayCode(n: int):
        '''
        生成n位格雷码
        
        参数:
            n: 格雷码位数
            
        返回:
            list: 格雷码列表
        '''
        if n < 1:
            print("输入数字必须大于0")
        else:
            code = ["0", "1"]
            for i in range(1, n):  # 循环递归
                code_lift = ["0" + idx for idx in code]  # 在前面添加0
                code_right = ["1" + idx for idx in code[::-1]]  # 在前面添加1，并反转列表
                code = code_lift + code_right  # 合并两个列表
            return code

    def __formCodes(self, n: int):
        '''
        生成codes矩阵
        
        将格雷码转换为矩阵形式，便于后续处理
        
        参数:
            n: 格雷码位数
            
        返回:
            numpy.ndarray: 格雷码矩阵
        '''
        code_temp = GrayCode.__createGrayCode(n)       # 首先生成n位格雷码储存在code_temp中
        codes = []
        for row in range(len(code_temp[0])):           # n位格雷码循环n次
            c = []
            for idx in range(len(code_temp)):          # 循环2**n次
                c.append(int(code_temp[idx][row]))     # 将code_temp中第idx个元素中的第row个数添加到c中
            codes.append(c)
        return np.array(codes, np.uint8)

    def toPattern(self, idx: int, cols: int = 1920, rows: int = 1080):
        '''
        生成垂直条纹格雷码光栅图
        
        将格雷码转换为投影用的垂直条纹光栅图案
        
        参数:
            idx: 格雷码索引
            cols: 图像宽度，默认1920
            rows: 图像高度，默认1080
            
        返回:
            numpy.ndarray: 格雷码光栅图像
        '''
        row = self.codes[idx, :]
        one_row = np.zeros((cols), np.uint8)
        per_col = int(cols / len(row))
        for i in range(len(row)):
            one_row[i * per_col: (i + 1) * per_col] = row[i]
        pattern = np.tile(one_row, (rows, 1)) * 255
        return pattern
        
    def toHorizontalPattern(self, idx: int, cols: int = 1920, rows: int = 1080):
        '''
        生成水平条纹格雷码光栅图
        
        将格雷码转换为投影用的水平条纹光栅图案
        
        参数:
            idx: 格雷码索引
            cols: 图像宽度，默认1920
            rows: 图像高度，默认1080
            
        返回:
            numpy.ndarray: 格雷码光栅图像
        '''
        row = self.codes[idx, :]
        pattern = np.zeros((rows, cols), np.uint8)
        per_row = int(rows / len(row))
        for i in range(len(row)):
            pattern[i * per_row: (i + 1) * per_row, :] = row[i] * 255
        return pattern

    def __code2k(self, k):
        '''
        将k映射到对应的格雷码
        
        参数:
            k: 索引值
            
        返回:
            str: 对应的格雷码字符串
        '''
        col = self.codes[:, k]
        code = ""
        for i in col:
            code += str(i)
        return code

    def __k2v(self, k):
        '''
        将k映射为v值（二进制转十进制）
        
        参数:
            k: 索引值
            
        返回:
            int: 对应的十进制值
        '''
        col = list(self.codes[:, k])
        col = [str(i) for i in col]
        code = "".join(col)
        v = int(code,2)
        return v


def generate_graycode_patterns(width=1024, height=768, direction="vertical", bits=5, 
                              noise_level=0, save_dir=None):
    """
    生成格雷码条纹图案
    
    参数:
        width: 图像宽度，默认1024
        height: 图像高度，默认768
        direction: 条纹方向，'horizontal'或'vertical'，默认'vertical'（垂直条纹）
        bits: 格雷码位数，默认5
        noise_level: 噪声水平，默认0（无噪声）
        save_dir: 保存目录，默认根据方向自动选择
    
    返回:
        list: 生成的格雷码图像列表
    """
    # 确定保存目录
    if save_dir is None:
        if direction.lower() == "horizontal":
            save_dir = "gray_patterns/gray_patterns_vertical"  # 水平条纹用于垂直方向解包裹
        else:
            save_dir = "gray_patterns/gray_patterns_horizontal"  # 垂直条纹用于水平方向解包裹
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    print(f"确保{save_dir}目录存在")
    
    # 创建格雷码生成器
    g = GrayCode(bits)
    
    # 生成格雷码图像
    images = []
    binary_images = []
    
    for i in range(bits):
        # 根据指定的方向生成条纹
        if direction.lower() == "horizontal":
            # 水平条纹（用于垂直方向解包裹）
            pattern = g.toHorizontalPattern(i, width, height)
            pattern_type = "水平条纹"
        else:
            # 垂直条纹（用于水平方向解包裹）
            pattern = g.toPattern(i, width, height)
            pattern_type = "垂直条纹"
        
        # 添加噪声（如果指定）
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, pattern.shape)
            pattern = np.clip(pattern + noise, 0, 255).astype(np.uint8)
        
        # 添加轻微高斯模糊（模拟光学系统）
        pattern_blurred = cv.GaussianBlur(pattern, (3, 3), 0.5)
        
        # 生成二值化图像
        _, binary_pattern = cv.threshold(pattern_blurred, 127, 255, cv.THRESH_BINARY)
        
        # 保存原始格雷码图案
        gray_filename = f"{save_dir}/gray_bit_{i}.png"
        cv.imwrite(gray_filename, pattern_blurred)
        
        # 保存二值化格雷码图案
        binary_filename = f"{save_dir}/matched_binary_{i}.png"
        cv.imwrite(binary_filename, binary_pattern)
        
        print(f"生成了{pattern_type}格雷码图像: {gray_filename} (位 {i+1}/{bits})")
        
        images.append(pattern_blurred)
        binary_images.append(binary_pattern)
    
    # 如果是垂直条纹，复制到gray_patterns目录（兼容性考虑）
    if direction.lower() == "vertical":
        copy_dir = "gray_patterns"
        os.makedirs(copy_dir, exist_ok=True)
        print(f"确保{copy_dir}目录存在")
        
        for i in range(bits):
            src_gray = f"{save_dir}/gray_bit_{i}.png"
            dst_gray = f"{copy_dir}/gray_bit_{i}.png"
            cv.imwrite(dst_gray, images[i])
            
            src_binary = f"{save_dir}/matched_binary_{i}.png"
            dst_binary = f"{copy_dir}/matched_binary_{i}.png"
            cv.imwrite(dst_binary, binary_images[i])
            
        print(f"已将垂直条纹格雷码图像复制到 {copy_dir} 目录")
    
    return images, binary_images

def show_direction_examples():
    """显示方向示例，帮助用户理解条纹方向与解包裹的关系"""
    print("\n格雷码条纹方向示例:")
    print("1. 垂直条纹 (用于水平方向解包裹)")
    print("   ┃┃┃┃┃┃┃┃  ← 条纹沿垂直方向延伸")
    print("   ┃┃┃┃┃┃┃┃  ← 相位沿水平方向变化")
    print("   ┃┃┃┃┃┃┃┃")
    print("   保存到: gray_patterns/gray_patterns_horizontal 和 gray_patterns")
    print("\n2. 水平条纹 (用于垂直方向解包裹)")
    print("   ━━━━━━━━  ← 条纹沿水平方向延伸")
    print("   ━━━━━━━━  ← 相位沿垂直方向变化")
    print("   ━━━━━━━━")
    print("   保存到: gray_patterns/gray_patterns_vertical")

def show_preview(direction, bits=5, width=30, height=10):
    """显示简单的ASCII预览图案"""
    print("\n预览 (ASCII简化版):")
    if direction == "horizontal":
        # 生成简化的水平条纹预览
        period = max(1, int(height / (2**(bits-3))))
        for i in range(height):
            if (i // period) % 2 == 0:
                print("  " + "━" * width)
            else:
                print("  " + " " * width)
    else:
        # 生成简化的垂直条纹预览
        period = max(1, int(width / (2**(bits-3))))
        for i in range(height):
            row = ""
            for j in range(width):
                if (j // period) % 2 == 0:
                    row += "┃"
                else:
                    row += " "
            print("  " + row)

def main():
    """主函数，处理命令行参数并生成格雷码条纹图案"""
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='生成格雷码条纹图案')
    parser.add_argument('--width', type=int, default=1024, help='图像宽度（像素）')
    parser.add_argument('--height', type=int, default=768, help='图像高度（像素）')
    parser.add_argument('--direction', type=str, choices=['horizontal', 'vertical'], 
                        default='vertical', help='条纹方向: horizontal (水平条纹) 或 vertical (垂直条纹)')
    parser.add_argument('--bits', type=int, default=5, help='格雷码位数')
    parser.add_argument('--noise', type=int, default=0, help='噪声水平（标准差）')
    parser.add_argument('--save_dir', type=str, default=None, help='保存目录（默认根据方向自动选择）')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 初始化变量，默认使用命令行参数的值
    width = args.width
    height = args.height
    direction = args.direction
    bits = args.bits
    noise_level = args.noise
    save_dir = args.save_dir
    
    # 如果是在交互环境中运行，提供详细的交互界面
    try:
        # 检测是否在交互环境或IDE中运行
        interactive_mode = True
        import sys
        interactive_mode = len(sys.argv) <= 1
    except:
        interactive_mode = True
    
    if interactive_mode:
        # 显示欢迎信息
        print("\n" + "="*50)
        print(" 格雷码条纹图案生成器 ")
        print("="*50)
        
        # 模式选择
        print("\n请选择生成模式:")
        print("1. 生成单个方向的格雷码图案")
        print("2. 一键生成两个方向的格雷码图案")
        mode_choice = input("请选择模式 (1/2，默认为1): ")
        
        if mode_choice == "2":
            # --- 批处理模式 ---
            print("\n--- 批处理模式: 生成两个方向的图案 ---")
            width = int(input(f"请输入图像宽度（像素，默认{args.width}）: ") or args.width)
            height = int(input(f"请输入图像高度（像素，默认{args.height}）: ") or args.height)
            bits = int(input(f"请输入格雷码位数（默认{args.bits}）: ") or args.bits)
            noise_level = int(input(f"请输入噪声水平（默认{args.noise}，通常为0）: ") or args.noise)
            
            print("\n准备生成两个方向的格雷码图案...")
            confirm = input("确认生成? (y/n，默认y): ").lower()
            if confirm == "n":
                print("操作已取消")
                return
            
            # 生成垂直条纹
            print("\n--- 正在生成垂直条纹 ---")
            generate_graycode_patterns(width=width, height=height, direction='vertical', 
                                     bits=bits, noise_level=noise_level, save_dir=None)
            
            # 生成水平条纹
            print("\n--- 正在生成水平条纹 ---")
            generate_graycode_patterns(width=width, height=height, direction='horizontal', 
                                     bits=bits, noise_level=noise_level, save_dir=None)
            
            print("\n所有格雷码条纹图案生成完成！")
            return
            
        # --- 单方向模式 ---
        # 显示条纹方向的图形示例
        show_direction_examples()
        
        print("\n请根据您的扫描需求选择合适的条纹方向:")
        print("- 如需解算水平方向相位，请选择垂直条纹")
        print("- 如需解算垂直方向相位，请选择水平条纹")
        
        # 用户选择条纹方向
        while True:
            direction_choice = input("\n请选择条纹方向 (1=垂直条纹, 2=水平条纹, 默认为1): ")
            if direction_choice == "2":
                direction = "horizontal"
                print("已选择: 水平条纹（用于垂直方向解包裹）")
                break
            elif direction_choice == "1" or direction_choice == "":
                direction = "vertical"
                print("已选择: 垂直条纹（用于水平方向解包裹）")
                break
            else:
                print("无效选择，请输入1或2")
        
        # 用户输入图像参数
        print("\n请设置图像参数:")
        width = int(input(f"请输入图像宽度（像素，默认{args.width}）: ") or args.width)
        height = int(input(f"请输入图像高度（像素，默认{args.height}）: ") or args.height)
        
        bits = int(input(f"请输入格雷码位数（默认{args.bits}）: ") or args.bits)
        show_preview(direction, bits)
        
        # 允许用户调整位数直到满意
        while True:
            adjust = input("是否调整格雷码位数? (y/n，默认n): ").lower()
            if adjust == "y":
                bits = int(input(f"请输入新的格雷码位数（当前{bits}）: ") or bits)
                show_preview(direction, bits)
            else:
                break
        
        # 获取剩余参数
        noise_level = int(input(f"请输入噪声水平（默认{args.noise}，通常为0）: ") or args.noise)
        
        # 确定保存目录
        if direction.lower() == "horizontal":
            default_dir = "gray_patterns/gray_patterns_vertical"  # 水平条纹用于垂直方向解包裹
        else:
            default_dir = "gray_patterns/gray_patterns_horizontal"  # 垂直条纹用于水平方向解包裹
            
        save_dir = input(f"请输入保存目录（默认 {default_dir}）: ") or default_dir
        
        # 确认生成
        print("\n准备生成格雷码条纹图案:")
        print(f"- 图像尺寸: {width}x{height}像素")
        print(f"- 条纹方向: {'水平' if direction == 'horizontal' else '垂直'}")
        print(f"- 格雷码位数: {bits}")
        print(f"- 噪声水平: {noise_level}")
        print(f"- 保存目录: {save_dir}")
        
        confirm = input("\n确认生成? (y/n，默认y): ").lower()
        if confirm == "n":
            print("操作已取消")
            return
    
    # 生成格雷码条纹图案
    generate_graycode_patterns(
        width=width, 
        height=height, 
        direction=direction, 
        bits=bits, 
        noise_level=noise_level, 
        save_dir=save_dir
    )
    
    print("\n格雷码条纹图案生成完成！")
    print(f"图像尺寸: {width}x{height}像素")
    print(f"条纹方向: {'水平' if direction == 'horizontal' else '垂直'}")
    print(f"格雷码位数: {bits}")
    
    if direction == "horizontal":
        print("\n提示：生成的水平条纹格雷码图案适用于垂直方向解包裹")
        print(f"图像已保存到 {save_dir or 'gray_patterns/gray_patterns_vertical'} 目录")
    else:
        print("\n提示：生成的垂直条纹格雷码图案适用于水平方向解包裹")
        print(f"图像已保存到 {save_dir or 'gray_patterns/gray_patterns_horizontal'} 和 gray_patterns 目录")
    
    print("使用这些图案时，请将其投影到物体表面，然后用相机采集反射图像")

if __name__ == "__main__":
    main() 