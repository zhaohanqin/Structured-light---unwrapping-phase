import numpy as np
import cv2 as cv
import os
import math
import argparse

def generate_fringe_patterns(width=1024, height=768, direction="vertical", frequency=15, 
                            intensity=100, offset=128, noise_level=0, save_dir="fringe_patterns", steps=4):
    """
    生成N步相移条纹图案
    
    参数:
        width: 图像宽度，默认1024
        height: 图像高度，默认768
        direction: 条纹方向，'horizontal'或'vertical'，默认'vertical'（垂直条纹）
        frequency: 条纹频率，默认15
        intensity: 条纹强度，默认100（振幅）
        offset: 亮度偏移，默认128（平均灰度值）
        noise_level: 噪声水平，默认0（无噪声）
        save_dir: 保存目录，默认'fringe_patterns'
        steps: 相移步数，默认4
    
    返回:
        list: 生成的N步相移图像列表
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建了{save_dir}目录")
    
    # 创建网格
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    
    # 生成N步相移图像
    images = []
    
    for i in range(steps):
        # 计算相移量: 2π/N * i
        phase_shift = i * 2 * math.pi / steps
        
        # 根据指定的方向生成条纹
        if direction.lower() == "horizontal":
            # 水平条纹（沿水平方向延伸，相位沿垂直方向变化）
            fringe = offset + intensity * np.sin(2 * math.pi * frequency * y + phase_shift)
            pattern_type = "水平条纹"
            filename_prefix = "I" + str(i + 1 + steps)  # I(N+1) to I(2N)
        else:
            # 垂直条纹（沿垂直方向延伸，相位沿水平方向变化）
            fringe = offset + intensity * np.sin(2 * math.pi * frequency * x + phase_shift)
            pattern_type = "垂直条纹"
            filename_prefix = "I" + str(i + 1)  # I1 to IN
        
        # 添加噪声（如果指定）
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, fringe.shape)
            fringe = fringe + noise
        
        # 确保像素值在[0, 255]范围内
        fringe = np.clip(fringe, 0, 255).astype(np.uint8)
        
        # 保存图像
        filename = f"{save_dir}/{filename_prefix}.png"
        cv.imwrite(filename, fringe)
        phase_degrees = int(phase_shift * 180 / math.pi)
        print(f"生成了{pattern_type}图像: {filename} (第{i+1}/{steps}步, 相移: {phase_degrees}°)")
        
        images.append(fringe)
    
    return images

def show_direction_examples():
    """显示方向示例，帮助用户理解条纹方向"""
    print("\n条纹方向示例 (N步相移):")
    print("1. 水平条纹 (用于垂直方向解包裹)")
    print("   ━━━━━━━━  ← 条纹沿水平方向延伸")
    print("   ━━━━━━━━  ← 相位沿垂直方向变化")
    print("   ━━━━━━━━")
    print("   生成文件：I(N+1).png ... I(2N).png")
    print("\n2. 垂直条纹 (用于水平方向解包裹)")
    print("   ┃┃┃┃┃┃┃┃  ← 条纹沿垂直方向延伸")
    print("   ┃┃┃┃┃┃┃┃  ← 相位沿水平方向变化")
    print("   ┃┃┃┃┃┃┃┃")
    print("   生成文件：I1.png ... IN.png")

def show_preview(direction, frequency, width=30, height=10):
    """显示简单的ASCII预览图案"""
    print("\n预览 (ASCII简化版):")
    if direction == "horizontal":
        # 生成简化的水平条纹预览
        period = max(1, int(height / (frequency/3)))
        for i in range(height):
            if (i // (period//2)) % 2 == 0:
                print("  " + "━" * width)
            else:
                print("  " + " " * width)
    else:
        # 生成简化的垂直条纹预览
        period = max(1, int(width / (frequency/3)))
        for i in range(height):
            row = ""
            for j in range(width):
                if (j // (period//2)) % 2 == 0:
                    row += "┃"
                else:
                    row += " "
            print("  " + row)

def main():
    """主函数，处理命令行参数并生成N步相移图案"""
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='生成N步相移条纹图案')
    parser.add_argument('--width', type=int, default=1024, help='图像宽度（像素）')
    parser.add_argument('--height', type=int, default=768, help='图像高度（像素）')
    parser.add_argument('--direction', type=str, choices=['horizontal', 'vertical'], 
                        default='vertical', help='条纹方向: horizontal (水平条纹) 或 vertical (垂直条纹)')
    parser.add_argument('--frequency', type=int, default=15, help='条纹频率（每单位长度的条纹数）')
    parser.add_argument('--intensity', type=int, default=100, help='条纹强度（振幅）')
    parser.add_argument('--offset', type=int, default=128, help='亮度偏移（平均灰度值）')
    parser.add_argument('--noise', type=int, default=0, help='噪声水平（标准差）')
    parser.add_argument('--save_dir', type=str, default='fringe_patterns', help='保存目录')
    parser.add_argument('--steps', type=int, default=4, help='相移步数 (N)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 初始化变量，默认使用命令行参数的值
    width = args.width
    height = args.height
    direction = args.direction
    frequency = args.frequency
    intensity = args.intensity
    offset = args.offset
    noise_level = args.noise
    save_dir = args.save_dir
    steps = args.steps
    
    # 如果是在交互环境中运行，提供详细的交互界面
    try:
        # 检测是否在交互环境或IDE中运行
        interactive_mode = os.path.basename(os.getcwd()) == os.path.dirname(args.save_dir)
        if not interactive_mode:
            # 尝试检测是否有命令行参数
            import sys
            interactive_mode = len(sys.argv) <= 1
    except:
        interactive_mode = True
    
    if interactive_mode:
        # 显示欢迎信息
        print("\n" + "="*50)
        print(" N步相移条纹图案生成器 ")
        print("="*50)
        
        # 模式选择
        print("\n请选择生成模式:")
        print("1. 生成单个方向的相移图案")
        print("2. 一键生成两个方向的相移图案")
        mode_choice = input("请选择模式 (1/2，默认为2): ")

        if mode_choice == "1":
            # --- 单方向模式 ---
            show_direction_examples()
            print("\n请根据您的扫描需求选择合适的条纹方向:")
            print("- 如需解算水平方向相位，请选择垂直条纹 (将生成I1-IN)")
            print("- 如需解算垂直方向相位，请选择水平条纹 (将生成I(N+1)-I(2N))")
            
            # 用户选择条纹方向
            while True:
                direction_choice = input("\n请选择条纹方向 (1=水平条纹, 2=垂直条纹, 默认为2): ")
                if direction_choice == "1":
                    direction = "horizontal"
                    print("已选择: 水平条纹（用于垂直方向解包裹，将生成I(N+1)-I(2N)）")
                    break
                elif direction_choice == "2" or direction_choice == "":
                    direction = "vertical"
                    print("已选择: 垂直条纹（用于水平方向解包裹，将生成I1-IN）")
                    break
                else:
                    print("无效选择，请输入1或2")
        else:
            # --- 批处理模式 (默认) ---
            print("\n--- 批处理模式: 生成两个方向的图案 ---")
            direction = "both" # 特殊标记

        # 用户输入图像参数
        print("\n请设置图像参数:")
        steps = int(input(f"请输入相移步数N（默认{args.steps}）: ") or args.steps)
        width = int(input(f"请输入图像宽度（像素，默认{args.width}）: ") or args.width)
        height = int(input(f"请输入图像高度（像素，默认{args.height}）: ") or args.height)
        
        frequency = int(input(f"请输入条纹频率（默认{args.frequency}）: ") or args.frequency)
        show_preview(direction, frequency)
        
        # 允许用户调整频率直到满意
        while True:
            adjust = input("是否调整条纹频率? (y/n，默认n): ").lower()
            if adjust == "y":
                frequency = int(input(f"请输入新的条纹频率（当前{frequency}）: ") or frequency)
                show_preview(direction, frequency)
            else:
                break
        
        # 获取剩余参数
        intensity = int(input(f"请输入条纹强度（振幅，默认{args.intensity}）: ") or args.intensity)
        offset = int(input(f"请输入亮度偏移（平均灰度，默认{args.offset}）: ") or args.offset)
        noise_level = int(input(f"请输入噪声水平（默认{args.noise}，通常为0）: ") or args.noise)
        
        # 确认生成
        print(f"\n准备生成{steps}步相移图案:")
        print(f"- 图像尺寸: {width}x{height}像素")
        print(f"- 条纹方向: {'水平' if direction == 'horizontal' else '垂直'}")
        print(f"- 条纹频率: {frequency}")
        print(f"- 条纹强度: {intensity}")
        print(f"- 亮度偏移: {offset}")
        print(f"- 噪声水平: {noise_level}")
        print(f"- 保存目录: {save_dir}")
        
        confirm = input("\n确认生成? (y/n，默认y): ").lower()
        if confirm == "n":
            print("操作已取消")
            return
    
    # 生成条纹图案
    if direction == "both":
        # --- 批处理执行 ---
        print("\n--- 正在生成垂直条纹 (I1-IN) ---")
        generate_fringe_patterns(width, height, "vertical", frequency, intensity, offset, noise_level, save_dir, steps)
        
        print("\n--- 正在生成水平条纹 (I(N+1)-I(2N)) ---")
        generate_fringe_patterns(width, height, "horizontal", frequency, intensity, offset, noise_level, save_dir, steps)
        
        print(f"\n所有 {steps}步相移图案生成完成！")
        print(f"图像已全部保存到 {save_dir} 目录")

    else:
        # --- 单方向执行 ---
        generate_fringe_patterns(
            width=width, 
            height=height, 
            direction=direction, 
            frequency=frequency, 
            intensity=intensity, 
            offset=offset, 
            noise_level=noise_level, 
            save_dir=save_dir,
            steps=steps
        )
        
        print(f"\n{steps}步相移图案生成完成！")
        print(f"图像尺寸: {width}x{height}像素")
        print(f"条纹方向: {'水平' if direction == 'horizontal' else '垂直'}")
        print(f"条纹频率: {frequency}")
        print(f"相移图像已保存到 {save_dir} 目录")
        if direction == "horizontal":
            print(f"\n提示：生成的水平条纹图案(I{steps+1}-I{2*steps})适用于垂直方向解包裹")
        else:
            print(f"\n提示：生成的垂直条纹图案(I1-I{steps})适用于水平方向解包裹")
    
    print("使用这些图案时，请将其投影到物体表面，然后用相机采集反射图像")

if __name__ == "__main__":
    main() 