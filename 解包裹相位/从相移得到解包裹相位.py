import os
import glob
import cv2 as cv
import numpy as np
import argparse
import re
from wrapped_phase import WrappedPhase
from phase_unwrapper import PhaseUnwrapper, generate_combined_phase_image

def load_images_from_folder(folder_path, expected_count=None):
    """从文件夹加载图像文件"""
    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹不存在 - {folder_path}")
        return None, None

    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    image_paths = []
    
    # 优先使用数字排序
    def extract_number(path):
        match = re.search(r'(\d+)', os.path.basename(path))
        return int(match.group(1)) if match else float('inf')

    all_files = os.listdir(folder_path)
    all_files.sort(key=lambda f: extract_number(os.path.join(folder_path, f)))

    for file_name in all_files:
        if file_name.lower().endswith(tuple(image_extensions)):
            image_paths.append(os.path.join(folder_path, file_name))

    if not image_paths:
        print(f"错误: 在文件夹 {folder_path} 中未找到图像")
        return None, None

    if expected_count and len(image_paths) < expected_count:
        print(f"警告: 在 {folder_path} 中找到 {len(image_paths)} 张图像, 但期望 {expected_count} 张。")
    elif expected_count and len(image_paths) > expected_count:
        print(f"警告: 在 {folder_path} 中找到 {len(image_paths)} 张图像, 超过期望的 {expected_count} 张。将使用前 {expected_count} 张。")
        image_paths = image_paths[:expected_count]

    images = []
    basenames = []
    for path in image_paths:
        img = cv.imread(path, -1)
        if img is not None:
            # 确保图像是灰度图
            if len(img.shape) > 2:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            images.append(img)
            basenames.append(os.path.splitext(os.path.basename(path))[0])
        else:
            print(f"警告: 无法读取图像 {path}")

    print(f"从 {folder_path} 成功加载 {len(images)} 张图像。")
    return images, basenames

def load_and_split_fringe_images(folder_path, steps):
    """从单个文件夹加载并分割水平和垂直相移图像"""
    if not os.path.isdir(folder_path):
        print(f"错误: 相移图像文件夹不存在 - {folder_path}")
        return None, None

    total_expected = 2 * steps
    print(f"\n从 {folder_path} 加载 {total_expected} 张相移图像 ({steps}张用于水平解包, {steps}张用于垂直解包)...")

    all_images, _ = load_images_from_folder(folder_path, expected_count=total_expected)

    if not all_images or len(all_images) < total_expected:
        print(f"错误: 文件夹中的相移图像数量不足。需要 {total_expected} 张，但只找到 {len(all_images) if all_images else 0} 张。")
        return None, None

    # 根据用户最终要求：
    # I1-IN 是垂直条纹 (用于水平解包裹)
    # I(N+1)-I2N 是水平条纹 (用于垂直解包裹)
    images_for_h_unwrap = all_images[0:steps]
    images_for_v_unwrap = all_images[steps:2*steps]

    print(f"成功分割图像: {len(images_for_h_unwrap)} 张用于水平解包, {len(images_for_v_unwrap)} 张用于垂直解包。")
    return images_for_h_unwrap, images_for_v_unwrap

def run_full_pipeline(args):
    """
    执行完整的3D重建流程
    """
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"创建输出目录: {args.output}")

    unwrapped_phase_h, unwrapped_phase_v = None, None

    # --- 执行水平解包裹流程 ---
    if args.mode in ['horizontal', 'both']:
        print("\n" + "="*40)
        print("  执行水平方向解包裹流程")
        print("="*40)
        
        # 1. 加载图像
        images_for_h_unwrap, _ = load_and_split_fringe_images(args.fringes, args.steps)
        v_gray_images, _ = load_images_from_folder(args.v_graycodes, expected_count=args.gray_bits)

        if images_for_h_unwrap and v_gray_images:
            # 2. 计算包裹相位
            wp_h = WrappedPhase(n=args.steps)
            wrapped_phase_h = wp_h.computeWrappedphase(images_for_h_unwrap)
            wp_h.save_wrapped_phase(wrapped_phase_h, args.output, "h_unwrap_from_v_fringe_", direction="vertical")
            
            # 3. 解包裹相位
            unwrapper_h = PhaseUnwrapper(n=args.gray_bits, direction="horizontal")
            # 仅在单方向模式下显示中间结果
            show_h_results = (args.mode == 'horizontal')
            unwrapped_phase_h = unwrapper_h.unwrap_phase(
                wrapped_phase_h, v_gray_images, 
                show_results=show_h_results, 
                save_results=True,  # 总是保存结果
                basename="horizontal_unwrapped"
            )
        else:
            print("错误：水平解包裹所需图像不完整，跳过此流程。")

    # --- 执行垂直解包裹流程 ---
    if args.mode in ['vertical', 'both']:
        print("\n" + "="*40)
        print("  执行垂直方向解包裹流程")
        print("="*40)

        # 1. 加载图像
        _, images_for_v_unwrap = load_and_split_fringe_images(args.fringes, args.steps)
        h_gray_images, _ = load_images_from_folder(args.h_graycodes, expected_count=args.gray_bits)

        if images_for_v_unwrap and h_gray_images:
            # 2. 计算包裹相位
            wp_v = WrappedPhase(n=args.steps)
            wrapped_phase_v = wp_v.computeWrappedphase(images_for_v_unwrap)
            wp_v.save_wrapped_phase(wrapped_phase_v, args.output, "v_unwrap_from_h_fringe_", direction="horizontal")

            # 3. 解包裹相位
            unwrapper_v = PhaseUnwrapper(n=args.gray_bits, direction="vertical")
            # 仅在单方向模式下显示中间结果
            show_v_results = (args.mode == 'vertical')
            unwrapped_phase_v = unwrapper_v.unwrap_phase(
                wrapped_phase_v, h_gray_images, 
                show_results=show_v_results, 
                save_results=True,  # 总是保存结果
                basename="vertical_unwrapped"
            )
        else:
            print("错误：垂直解包裹所需图像不完整，跳过此流程。")

    # --- 合并结果 (仅在 both 模式下) ---
    if args.mode == 'both':
        print("\n" + "="*40)
        print("  步骤 3: 合并结果并可视化")
        print("="*40)
        if unwrapped_phase_h is not None and unwrapped_phase_v is not None:
            output_path = os.path.join(args.output, "final_combined_phase.png")
            print(f"\n生成组合相位图: {output_path}")
            generate_combined_phase_image(unwrapped_phase_h, unwrapped_phase_v, output_path)
        else:
            print("错误: 一个或两个方向的解包裹失败，无法生成组合图像。")
        
    print("\n" + "="*40)
    print("  🎉 全部流程处理完成! 🎉")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="完整的结构光三维重建流程，从相移、格雷码图像到最终的解包裹相位。\n"
                    "可以直接通过命令行参数运行，或者不带参数进入交互式设置模式。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- 模式选择 ---
    parser.add_argument('--mode', type=str, choices=['horizontal', 'vertical', 'both'], 
                        default=None, help='选择要执行的解包裹方向。')

    # --- 输入路径参数 (不再强制要求) ---
    parser.add_argument('--fringes', type=str, help='包含所有相移图像(水平和垂直)的文件夹路径。')
    parser.add_argument('--v_graycodes', type=str, help='包含投影[垂直条纹]格雷码后拍摄的图像的文件夹路径。')
    parser.add_argument('--h_graycodes', type=str, help='包含投影[水平条纹]格雷码后拍摄的图像的文件夹路径。')
    
    # --- 算法参数 ---
    parser.add_argument('--steps', type=int, default=4, help='相移步数 (例如: 4 代表四步相移)。默认为4。')
    parser.add_argument('--gray_bits', type=int, default=5, help='格雷码的位数。默认为5。')
    
    # --- 输出参数 ---
    parser.add_argument('--output', type=str, default='reconstruction_results', help="所有结果的输出根目录。默认为 'reconstruction_results'。")
    
    args = parser.parse_args()

    # --- 检查是否进入交互模式 ---
    # 如果未指定模式，或者指定了模式但缺少必要的路径，则进入交互模式
    is_interactive = args.mode is None
    if not is_interactive:
        if args.mode in ['horizontal', 'both'] and (not args.fringes or not args.v_graycodes):
            is_interactive = True
        if args.mode in ['vertical', 'both'] and (not args.fringes or not args.h_graycodes):
            is_interactive = True

    if is_interactive:
        print("\n" + "="*50)
        print("--- 欢迎进入交互式设置模式 ---")
        print("="*50)

        # 辅助函数，用于获取有效的文件夹路径
        def get_valid_path(prompt_text):
            while True:
                path = input(prompt_text).strip().replace("'", "").replace('"', '')
                if os.path.isdir(path):
                    return path
                else:
                    print(f"  [错误] 路径 '{path}' 无效或不是一个文件夹，请重新输入。")

        # 1. 选择模式
        if not args.mode:
            while True:
                print("\n请选择要执行的解包裹模式:")
                print("  1. 仅水平方向 (Horizontal)")
                print("  2. 仅垂直方向 (Vertical)")
                print("  3. 两个方向并合并 (Both)")
                mode_choice = input("请输入您的选择 (1/2/3): ").strip()
                if mode_choice == '1':
                    args.mode = 'horizontal'
                    break
                elif mode_choice == '2':
                    args.mode = 'vertical'
                    break
                elif mode_choice == '3':
                    args.mode = 'both'
                    break
                else:
                    print("  [错误] 无效输入，请输入1, 2, 或 3。")

        # 2. 根据模式请求相应路径
        print(f"\n--- 已选择模式: {args.mode.upper()} ---")
        print("--- 请输入所需图像的文件夹路径 ---")
        
        if args.mode in ['horizontal', 'both'] and not args.fringes:
            args.fringes = get_valid_path("请输入包含[所有相移图像]的文件夹路径: ")
        if args.mode in ['horizontal', 'both'] and not args.v_graycodes:
            args.v_graycodes = get_valid_path("请输入[垂直条纹]格雷码图像的文件夹路径: ")

        # 如果是vertical模式且fringes未提供，也需要输入
        if args.mode == 'vertical' and not args.fringes:
             args.fringes = get_valid_path("请输入包含[所有相移图像]的文件夹路径: ")
        if args.mode in ['vertical', 'both'] and not args.h_graycodes:
            args.h_graycodes = get_valid_path("请输入[水平条纹]格雷码图像的文件夹路径: ")
        
        print("\n--- 路径设置完成 ---")
        print("\n--- 请设置算法参数 (可直接按Enter使用默认值) ---")

        # 辅助函数，用于获取有效的整数输入
        def get_valid_int(prompt_text, default_val, min_val=1):
            while True:
                val_str = input(f"{prompt_text} (默认为 {default_val}): ").strip()
                if not val_str:
                    return default_val
                try:
                    val_int = int(val_str)
                    if val_int >= min_val:
                        return val_int
                    else:
                        print(f"  [错误] 输入值必须大于或等于 {min_val}。")
                except ValueError:
                    print("  [错误] 请输入一个有效的整数。")

        args.steps = get_valid_int("请输入[相移步数]", default_val=args.steps, min_val=3)
        args.gray_bits = get_valid_int("请输入[格雷码位数]", default_val=args.gray_bits, min_val=3)

        print("\n--- 参数设置完成，开始执行重建流程 ---")
    
    # 运行主流程
    run_full_pipeline(args) 