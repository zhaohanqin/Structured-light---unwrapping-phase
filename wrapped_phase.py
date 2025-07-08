import numpy as np
import cv2 as cv
import math
import os
import argparse
import glob

# 添加图像尺寸标准化的工具函数
def normalize_image_size(images, target_size=None, method="crop"):
    """
    将一组图像标准化为相同的尺寸
    
    参数:
        images: 单张图像或图像列表
        target_size: 目标尺寸(height, width)，如果为None则使用所有图像的最小尺寸
        method: 尺寸调整方法，可选值为"crop"(裁剪)或"resize"(缩放)
        
    返回:
        标准化后的图像或图像列表
    """
    # 处理单张图像的情况
    if not isinstance(images, list):
        return images
    
    # 如果列表为空，直接返回
    if len(images) == 0:
        return images
    
    # 过滤掉None值和非图像对象
    valid_images = []
    for img in images:
        if img is not None:
            try:
                # 检查是否有shape属性和至少2个维度
                if hasattr(img, 'shape') and len(img.shape) >= 2:
                    valid_images.append(img)
                else:
                    print(f"警告: 跳过无效图像对象，shape={getattr(img, 'shape', 'unknown')}")
            except Exception as e:
                print(f"警告: 处理图像对象时出错: {e}")
    
    if len(valid_images) == 0:
        print("错误: 没有有效的图像可供处理")
        return images
    
    # 如果未指定目标尺寸，计算所有图像的最小尺寸
    if target_size is None:
        try:
            min_height = min(img.shape[0] for img in valid_images)
            min_width = min(img.shape[1] for img in valid_images)
            target_size = (min_height, min_width)
        except Exception as e:
            print(f"计算最小尺寸时出错: {e}")
            # 使用第一个有效图像的尺寸作为默认值
            target_size = valid_images[0].shape[:2]
            print(f"使用第一个有效图像的尺寸作为目标尺寸: {target_size}")
    
    # 检查图像是否需要调整
    need_adjustment = False
    for img in valid_images:
        if img.shape[0] != target_size[0] or img.shape[1] != target_size[1]:
            need_adjustment = True
            break
    
    # 如果所有图像尺寸已经一致且符合目标尺寸，直接返回
    if not need_adjustment:
        return images
    
    # 调整图像尺寸
    result = []
    for i, img in enumerate(images):
        if img is None:
            result.append(None)
            continue
        
        try:
            # 确保图像有正确的维度
            if not hasattr(img, 'shape') or len(img.shape) < 2:
                print(f"警告: 跳过第{i+1}个无效图像对象")
                result.append(None)
                continue
                
            if method == "crop":
                # 确保不会超出图像边界
                if img.shape[0] >= target_size[0] and img.shape[1] >= target_size[1]:
                    # 裁剪到目标尺寸（从左上角开始）
                    adjusted_img = img[:target_size[0], :target_size[1]]
                else:
                    # 如果图像太小，先放大再裁剪
                    temp = cv.resize(img, (max(target_size[1], img.shape[1]), max(target_size[0], img.shape[0])))
                    adjusted_img = temp[:target_size[0], :target_size[1]]
                    print(f"警告: 图像{i+1}太小，已调整大小后裁剪")
            elif method == "resize":
                # 缩放到目标尺寸
                adjusted_img = cv.resize(img, (target_size[1], target_size[0]))
            else:
                raise ValueError(f"不支持的调整方法: {method}")
            
            result.append(adjusted_img)
        except Exception as e:
            print(f"处理第{i+1}个图像时出错: {e}")
            # 如果处理失败，添加None
            result.append(None)
    
    # 检查是否有足够的有效图像
    valid_result = [img for img in result if img is not None]
    if len(valid_result) == 0:
        print("错误: 所有图像处理后均无效")
    else:
        print(f"成功处理{len(valid_result)}/{len(images)}个图像")
    
    return result

class WrappedPhase():
    """
    包裹相位计算类
    
    该类实现了N步相移法计算包裹相位的核心算法。
    N步相移法通过投影N幅具有不同相移量的正弦条纹图案，
    然后根据采集到的N幅图像计算每个像素点的包裹相位值。
    """
    
    def __init__(self, n: int = 4):
        """
        初始化包裹相位计算器
        
        参数:
            n (int): 相移步数，默认为4（四步相移法）
        """
        self.n = n  # 相移步数
        # 保存标准尺寸，用于所有图像的标准化
        self.standard_size = None
        self.size_method = "crop"  # 默认使用裁剪方法
    
    @staticmethod
    def getImageData(image_paths=None, direction: str = "vertical", standard_size=None, size_method="crop", n: int = 4):
        """
        获取相机拍摄的n幅相移图像
        
        该方法从指定路径读取N步相移法采集的图像序列。
        如果找不到图像文件，会生成测试用的相移图像。
        
        参数:
            image_paths: 相移图像的路径列表，如果为None则使用默认路径
            direction (str): 条纹方向，可选值为"vertical"或"horizontal"，默认为"vertical"
            standard_size: 标准图像尺寸(height, width)，默认为None（自动计算）
            size_method: 尺寸调整方法，可选值为"crop"(裁剪)或"resize"(缩放)
            n (int): 相移步数，默认为4
            
        返回:
            list: 包含n幅相移图像的列表，所有图像具有相同的尺寸
        """
        I = []
        loaded_images = []  # 记录成功加载的图像
        default_height, default_width = 480, 640  # 默认测试图像大小
        
        # 如果未提供图像路径，使用默认路径
        if image_paths is None:
            image_paths = []
            for i in range(n):
                # 根据条纹方向选择不同的图像文件
                if direction == "vertical":
                    # 垂直条纹使用I1-In
                    image_paths.append(f"fringe_patterns/I{i+1}.png")
                else: # direction == "horizontal"
                    # 水平条纹使用I(n+1)-I2n
                    image_paths.append(f"fringe_patterns/I{i+n+1}.png")
        
        # 首先尝试读取所有图像，记录成功加载的图像
        for filename in image_paths:
            try:
                # 尝试直接读取图像
                img = cv.imread(filename, -1)
                if img is None:
                    # 如果直接读取失败，尝试使用二进制方式读取
                    img_file = np.fromfile(filename, dtype=np.uint8)
                    img = cv.imdecode(img_file, -1)
                
                if img is None:
                    raise FileNotFoundError(f"无法读取图像: {filename}")
                
                # 确保图像是灰度图
                if len(img.shape) > 2:
                    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    
                loaded_images.append(img)
                I.append(img)
                print(f"成功读取图像: {filename}, 尺寸: {img.shape}")
            except Exception as e:
                print(f"读取图像 {filename} 失败: {e}")
                print("将使用随机生成的测试图像代替...")
        
        # 确定测试图像的尺寸
        if loaded_images:
            # 如果有成功加载的图像，使用其尺寸作为测试图像的尺寸
            test_height, test_width = loaded_images[0].shape
        else:
            # 否则使用默认尺寸
            test_height, test_width = default_height, default_width
        
        # 生成缺失的图像
        while len(I) < n:
            print(f"生成第{len(I)+1}幅测试图像...")
            # 创建带有随机噪声的正弦图案
            x = np.arange(0, test_width)
            y = np.arange(0, test_height).reshape(test_height, 1)
            
            # 确定当前是第几幅图像
            phase = 2 * math.pi * len(I) / n  # 相移量
            freq = 10  # 频率
            
            if direction == "vertical":
                # 垂直条纹 (沿x轴变化)
                img = np.sin(2 * math.pi * freq * x / test_width + phase) * 127 + 128
            else: # direction == "horizontal"
                # 水平条纹 (沿y轴变化)
                img = np.sin(2 * math.pi * freq * y / test_height + phase) * 127 + 128
            
            img = img.astype(np.uint8)
            I.append(img)
        
        # 标准化所有图像的尺寸
        I = normalize_image_size(I, standard_size, size_method)
        
        # 打印图像尺寸信息
        if len(I) > 0 and I[0] is not None:
            print(f"相移图像尺寸: {I[0].shape[0]}x{I[0].shape[1]} ({direction}条纹)")
            print(f"相移步数: {n}")
        
        return I

    def computeWrappedphase(self, I):
        """
        计算包裹相位
        
        该方法实现了N步相移法的核心算法，通过N幅相移图像计算包裹相位。
        
        参数:
            I: N幅相移图像列表 [I0, I1, ..., I(N-1)]
               每幅图像的相移量为 2π*k/N，其中k为图像索引(0到N-1)
        
        返回:
            numpy.ndarray: 包裹相位矩阵，范围[0, 2π]
        """
        # 检查图像数量
        n = len(I)
        if n < 3:
            raise ValueError(f"至少需要3幅相移图像，但只提供了{n}幅")
        
        # 将所有图像转换为浮点数类型，提高计算精度
        I_float = [img.astype(np.float32) for img in I]
        
        # 获取图像尺寸
        height, width = I_float[0].shape
        
        # 初始化包裹相位矩阵
        pha = np.zeros((height, width), np.float32)
        
        # 根据相移步数选择合适的算法
        if n == 3:
            # 三步相移法
            print("使用三步相移法计算包裹相位...")
            # 三步相移法公式: φ = atan2(sqrt(3)*(I1-I3), 2*I2-I1-I3)
            for y in range(height):
                for x in range(width):
                    numerator = math.sqrt(3) * (I_float[0][y, x] - I_float[2][y, x])
                    denominator = 2 * I_float[1][y, x] - I_float[0][y, x] - I_float[2][y, x]
                    pha[y, x] = math.atan2(numerator, denominator)
        
        elif n == 4:
            # 四步相移法
            print("使用四步相移法计算包裹相位...")
            # 四步相移法公式: φ = atan2(I4-I2, I1-I3)
            for y in range(height):
                for x in range(width):
                    numerator = I_float[3][y, x] - I_float[1][y, x]
                    denominator = I_float[0][y, x] - I_float[2][y, x]
                    pha[y, x] = math.atan2(numerator, denominator)
        
        else:
            # N步相移法（通用算法）
            print(f"使用{n}步相移法计算包裹相位...")
            # N步相移法公式: φ = atan2(sum(I_k*sin(2πk/N)), sum(I_k*cos(2πk/N)))
            for y in range(height):
                for x in range(width):
                    numerator = 0.0
                    denominator = 0.0
                    for k in range(n):
                        phase_k = 2 * math.pi * k / n
                        numerator += I_float[k][y, x] * math.sin(phase_k)
                        denominator += I_float[k][y, x] * math.cos(phase_k)
                    pha[y, x] = math.atan2(numerator, denominator)
        
        # 确保相位在[0, 2π]范围内
        pha = np.where(pha < 0, pha + 2*math.pi, pha)
        
        return pha

    def save_wrapped_phase(self, wrapped_phase, output_dir="results", prefix="", direction="vertical"):
        """
        保存包裹相位结果
        
        参数:
            wrapped_phase: 包裹相位矩阵
            output_dir: 输出目录
            prefix: 文件名前缀
            direction: 条纹方向 ("vertical" 或 "horizontal")
        
        返回:
            str: 最终保存结果的完整目录路径
        """
        # 根据条纹方向确定子目录
        # 垂直条纹用于水平解包裹，水平条纹用于垂直解包裹
        if direction == "vertical":
            sub_dir = "wrapped_phase_horizontal"
        else:  # horizontal
            sub_dir = "wrapped_phase_vertical"
        
        final_output_dir = os.path.join(output_dir, sub_dir)
        
        # 确保输出目录存在
        if not os.path.exists(final_output_dir):
            os.makedirs(final_output_dir)
        
        # 将相位值缩放到[0,255]范围用于显示
        phase_scaled = (wrapped_phase * 255 / (2*math.pi)).astype(np.uint8)
        
        # 保存原始缩放视图
        cv.imwrite(os.path.join(final_output_dir, f"{prefix}wrapped_phase_original.png"), phase_scaled)
        
        # 应用伪彩色映射以增强可视化效果
        phase_color = cv.applyColorMap(phase_scaled, cv.COLORMAP_JET)
        cv.imwrite(os.path.join(final_output_dir, f"{prefix}wrapped_phase_color.png"), phase_color)
        
        # 应用直方图均衡化以增强对比度
        phase_eq = cv.equalizeHist(phase_scaled)
        cv.imwrite(os.path.join(final_output_dir, f"{prefix}wrapped_phase_equalized.png"), phase_eq)
        
        # 保存原始相位数据（numpy格式）
        np.save(os.path.join(final_output_dir, f"{prefix}wrapped_phase.npy"), wrapped_phase)
        
        print(f"包裹相位结果已保存到 {final_output_dir} 目录")
        return final_output_dir

    def visualize_wrapped_phase(self, wrapped_phase, title="包裹相位"):
        """
        可视化包裹相位
        
        参数:
            wrapped_phase: 包裹相位矩阵
            title: 显示窗口标题
        """
        # 将相位值缩放到[0,255]范围用于显示
        phase_scaled = (wrapped_phase * 255 / (2*math.pi)).astype(np.uint8)
        
        # 应用伪彩色映射以增强可视化效果
        phase_color = cv.applyColorMap(phase_scaled, cv.COLORMAP_JET)
        
        # 显示包裹相位
        cv.imshow(title, phase_color)
        
        # 定义鼠标回调函数，显示点击位置的相位值
        def mouse_callback(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                # 获取点击位置的相位值
                phase_value = wrapped_phase[y, x]
                # 在控制台显示信息
                print(f"点击位置 ({x}, {y}):")
                print(f"  相位值: {phase_value:.6f} rad")
                print(f"  相位角度: {phase_value * 180 / math.pi:.2f}°")
                
                # 在图像上显示相位值
                # 创建一个副本以避免修改原图
                display_img = phase_color.copy()
                # 绘制十字标记
                cv.drawMarker(display_img, (x, y), (0, 255, 255), cv.MARKER_CROSS, 20, 2)
                # 添加文本
                text = f"Phase: {phase_value:.2f} rad"
                cv.putText(display_img, text, (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                # 更新显示
                cv.imshow(title, display_img)
        
        # 设置鼠标回调
        cv.setMouseCallback(title, mouse_callback)
        
        print("\n交互提示:")
        print("- 鼠标点击图像可显示该点的相位值")
        print("- 按's'键保存图像")
        print("- 按'q'键或ESC键退出")
        
        # 等待用户按键
        while True:
            key = cv.waitKey(0)
            if key == ord("s"):
                # 按's'键保存图像
                if not os.path.exists("results"):
                    os.makedirs("results")
                cv.imwrite(f"results/wrapped_phase_user_saved.png", phase_color)
                print("已保存图像: results/wrapped_phase_user_saved.png")
            elif key == ord("q") or key == 27:  # 'q'键或ESC键
                break
        
        # 关闭窗口
        cv.destroyWindow(title)

def run_calculation(direction, steps, image_source, output_dir, size_method, target_size, visualize):
    """
    执行单次包裹相位计算的封装函数
    
    参数:
        direction (str): 条纹方向 ('vertical' 或 'horizontal')
        steps (int): 相移步数
        image_source (str or list): 图像文件夹路径或文件列表
        output_dir (str): 输出根目录
        size_method (str): 尺寸调整方法
        target_size (tuple): 目标尺寸
        visualize (bool): 是否可视化结果
    """
    print(f"\n{'='*20}")
    print(f"  开始计算: {'垂直条纹 (用于水平解包裹)' if direction == 'vertical' else '水平条纹 (用于垂直解包裹)'}")
    print(f"{'='*20}")

    wp = WrappedPhase(n=steps)
    wp.size_method = size_method
    wp.standard_size = target_size

    image_paths = None
    if isinstance(image_source, str) and os.path.isdir(image_source):
        # 从文件夹加载
        folder_path = image_source
        image_paths = []
        if direction == "vertical":
            file_names = [f"I{i+1}.png" for i in range(steps)]
        else:
            file_names = [f"I{i+steps+1}.png" for i in range(steps)]
        
        found_any = False
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            for i in range(steps):
                base_name = file_names[i].split('.')[0]
                file_path = os.path.join(folder_path, base_name + ext)
                if os.path.exists(file_path):
                    image_paths.append(file_path)
                    found_any = True
            if found_any: break

        if not image_paths:
            print(f"警告: 在 {folder_path} 中未找到 I* 格式的图像, 将尝试加载文件夹中所有图像...")
            all_images = []
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                all_images.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
            
            if all_images:
                def extract_number(path):
                    import re
                    match = re.search(r'(\d+)', os.path.basename(path))
                    return int(match.group(1)) if match else 0
                all_images.sort(key=extract_number)
                image_paths = all_images
    else:
        image_paths = image_source

    images = wp.getImageData(image_paths, direction, target_size, size_method, n=steps)

    valid_images = [img for img in images if img is not None]
    if len(valid_images) < 3:
        print(f"错误: 至少需要3幅相移图像，但只获取到{len(valid_images)}幅。跳过计算。")
        return

    print(f"使用{len(valid_images)}步相移法计算包裹相位...")
    wrapped_phase = wp.computeWrappedphase(images)

    print(f"包裹相位范围: [{np.min(wrapped_phase)}, {np.max(wrapped_phase)}]")
    
    prefix = "vertical_fringe_" if direction == "vertical" else "horizontal_fringe_"
    prefix += f"{len(valid_images)}step_"
    saved_path = wp.save_wrapped_phase(wrapped_phase, output_dir, prefix, direction=direction)
    print(f"结果已保存至: {os.path.abspath(saved_path)}")

    if visualize:
        title = f"{'垂直条纹' if direction == 'vertical' else '水平条纹'} - {len(valid_images)}步相移包裹相位"
        wp.visualize_wrapped_phase(wrapped_phase, title)

def main():
    """主函数，处理命令行参数并计算包裹相位"""
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='N步相移法计算包裹相位')
    parser.add_argument('--direction', type=str, choices=['vertical', 'horizontal'], 
                        default='vertical', help='条纹方向: vertical (垂直条纹) 或 horizontal (水平条纹)')
    parser.add_argument('--images', nargs='+', help='相移图像的路径（按顺序提供N幅图像）')
    parser.add_argument('--output', type=str, default='results', help='输出目录')
    parser.add_argument('--method', type=str, choices=['crop', 'resize'], 
                        default='crop', help='图像尺寸调整方法: crop (裁剪) 或 resize (缩放)')
    parser.add_argument('--size', nargs=2, type=int, help='目标图像尺寸 (宽 高)')
    parser.add_argument('--visualize', action='store_true', help='显示包裹相位可视化结果')
    parser.add_argument('--steps', type=int, default=4, help='相移步数，默认为4')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果通过命令行提供了图像，则直接处理并退出
    if args.images:
        target_size = (args.size[1], args.size[0]) if args.size else None
        run_calculation(args.direction, args.steps, args.images, args.output, 
                        args.method, target_size, args.visualize)
        return

    # --- 交互模式 ---
    print("\n请选择计算模式:")
    print("1. 计算单个方向的包裹相位")
    print("2. 一键计算两个方向的包裹相位")
    
    mode_choice = input("请选择模式 (1/2)，默认为1: ")
    if mode_choice == "2":
        # --- 批处理模式 ---
        print("\n--- 批处理模式: 计算两个方向的包裹相位 ---")
        
        steps_input = input(f"请输入相移步数 (默认为4): ")
        steps = int(steps_input) if steps_input and steps_input.isdigit() and int(steps_input) >= 3 else 4
        print(f"使用 {steps} 步相移法")

        v_fringe_folder = input("请输入[垂直条纹]图像所在的文件夹路径: ")
        h_fringe_folder = input("请输入[水平条纹]图像所在的文件夹路径: ")
        output_dir = input("请输入输出根目录 (默认为 'results'): ") or "results"
        
        # 处理垂直条纹
        if os.path.isdir(v_fringe_folder):
            run_calculation('vertical', steps, v_fringe_folder, output_dir, 'crop', None, False)
        else:
            print(f"错误: 路径无效，跳过垂直条纹计算: {v_fringe_folder}")
            
        # 处理水平条纹
        if os.path.isdir(h_fringe_folder):
            run_calculation('horizontal', steps, h_fringe_folder, output_dir, 'crop', None, False)
        else:
            print(f"错误: 路径无效，跳过水平条纹计算: {h_fringe_folder}")
            
        print("\n所有计算任务完成!")
        return

    # --- 单方向交互模式 (原有逻辑) ---
    wp = WrappedPhase(n=args.steps)
    
    # 设置图像尺寸调整方法
    wp.size_method = args.method
    
    # 设置目标尺寸（如果指定）
    if args.size:
        wp.standard_size = (args.size[1], args.size[0])  # 注意OpenCV使用(height, width)顺序
        print(f"目标图像尺寸: {args.size[0]}x{args.size[1]} (宽x高)")
    
    # 获取相移图像
    if args.images:
        if len(args.images) < 3:
            print(f"错误: 至少需要3幅相移图像，但只提供了{len(args.images)}幅")
            return
        elif len(args.images) != args.steps:
            print(f"警告: 指定的相移步数为{args.steps}，但提供了{len(args.images)}幅图像")
            print(f"将使用提供的{len(args.images)}幅图像作为实际相移步数")
            wp.n = len(args.images)
            
        print(f"使用用户提供的{len(args.images)}幅图像: {args.images}")
        images = wp.getImageData(args.images, args.direction, wp.standard_size, wp.size_method, n=len(args.images))
    else:
        # 使用交互式输入
        print("\n请选择投影的条纹方向:")
        print("1. 垂直条纹 (通常用于水平方向解包裹，加载 I1.png ~ In.png)")
        print("2. 水平条纹 (通常用于垂直方向解包裹，加载 I(n+1).png ~ I2n.png)")
        
        direction_choice = input("请选择条纹方向 (1/2)，默认为1: ")
        if direction_choice == "2":
            direction = "horizontal"
            print("已选择: 水平条纹")
        else:
            direction = "vertical"
            print("已选择: 垂直条纹")
        
        # 询问相移步数
        steps_input = input(f"请输入相移步数 (默认为{wp.n}): ")
        if steps_input and steps_input.isdigit():
            steps = int(steps_input)
            if steps < 3:
                print("错误: 相移步数至少为3，将使用默认值4")
                steps = 4
            wp.n = steps
        else:
            steps = wp.n
            
        print(f"使用{steps}步相移法")
        
        # 询问图像路径
        print("\n请输入图像路径:")
        
        # 检查默认路径是否存在
        default_exists = True
        default_paths = []
        for i in range(steps):
            if direction == "vertical":
                default_path = f"fringe_patterns/I{i+1}.png"
            else: # direction == "horizontal"
                default_path = f"fringe_patterns/I{i+steps+1}.png"
            
            default_paths.append(default_path)
            if not os.path.exists(default_path):
                default_exists = False
        
        # 询问是否使用自定义路径
        use_custom_path = True
        if default_exists:
            use_default = input(f"检测到默认路径下存在图像，是否使用默认路径? (y/n)，默认为y: ")
            if use_default.lower() != "n":
                use_custom_path = False
                image_paths = default_paths
                print(f"使用默认路径: {image_paths}")
        
        if use_custom_path:
            # 提示用户可以输入文件夹路径
            print("\n您可以直接输入图像所在的文件夹路径:")
            print("程序将自动查找文件夹中的图像文件:")
            print(f"- 垂直条纹: 查找I1.png到I{steps}.png")
            print(f"- 水平条纹: 查找I{steps+1}.png到I{2*steps}.png")
            print("\n或者使用通配符模式:")
            print("例如: E:/images/I*.png 将匹配 E:/images/I1.png, E:/images/I2.png 等")
            
            folder_or_pattern = input("请输入文件夹路径或图像路径模式 (留空使用测试图像): ")
            
            if folder_or_pattern:
                # 检查输入是否是文件夹路径
                if os.path.isdir(folder_or_pattern):
                    # 是文件夹路径，根据方向查找相应的图像文件
                    folder_path = folder_or_pattern
                    image_paths = []
                    
                    # 确保文件夹路径末尾有分隔符
                    if not folder_path.endswith(os.path.sep):
                        folder_path += os.path.sep
                    
                    # 根据方向选择要查找的文件名
                    if direction == "vertical":
                        # 垂直条纹，查找I1-In
                        file_names = [f"I{i+1}.png" for i in range(steps)]
                    else: # direction == "horizontal"
                        # 水平条纹，查找I(n+1)-I2n
                        file_names = [f"I{i+steps+1}.png" for i in range(steps)]
                    
                    # 检查文件是否存在
                    missing_files = []
                    for file_name in file_names:
                        file_path = os.path.join(folder_path, file_name)
                        if os.path.exists(file_path):
                            image_paths.append(file_path)
                        else:
                            missing_files.append(file_name)
                    
                    if missing_files:
                        print(f"警告: 在文件夹 {folder_path} 中未找到以下文件: {', '.join(missing_files)}")
                        
                        # 尝试查找其他格式的文件
                        for file_name in missing_files.copy():
                            base_name = file_name.split('.')[0]  # 去掉扩展名
                            # 尝试其他常见图像格式
                            for ext in ['.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                                alt_file_name = base_name + ext
                                alt_file_path = os.path.join(folder_path, alt_file_name)
                                if os.path.exists(alt_file_path):
                                    image_paths.append(alt_file_path)
                                    missing_files.remove(file_name)
                                    print(f"找到替代文件: {alt_file_name}")
                                    break
                        
                        # 如果仍有缺失文件，尝试查找数字命名的文件
                        if missing_files:
                            print("尝试查找数字命名的文件...")
                            for file_name in missing_files.copy():
                                # 从文件名中提取数字
                                import re
                                match = re.search(r'I(\d+)', file_name)
                                if match:
                                    num = match.group(1)
                                    # 尝试查找数字命名的文件
                                    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                                        alt_file_name = num + ext
                                        alt_file_path = os.path.join(folder_path, alt_file_name)
                                        if os.path.exists(alt_file_path):
                                            image_paths.append(alt_file_path)
                                            missing_files.remove(file_name)
                                            print(f"找到替代文件: {alt_file_name}")
                                            break
                    
                    # 对找到的文件进行排序，确保顺序正确
                    if image_paths:
                        # 尝试根据文件名中的数字进行排序
                        def extract_number(path):
                            match = re.search(r'(\d+)', os.path.basename(path))
                            if match:
                                return int(match.group(1))
                            return 0
                        
                        image_paths.sort(key=extract_number)
                    
                    # 如果找到了足够的文件
                    if len(image_paths) >= 3:
                        print(f"在文件夹 {folder_path} 中找到 {len(image_paths)} 个图像文件:")
                        for i, path in enumerate(image_paths):
                            print(f"  {i+1}. {os.path.basename(path)}")
                        
                        # 如果找到的文件数量不等于指定的步数，询问是否使用
                        if len(image_paths) != steps:
                            print(f"警告: 找到 {len(image_paths)} 个图像文件，但指定的相移步数为 {steps}")
                            use_found = input(f"是否使用这 {len(image_paths)} 个文件作为相移步数? (y/n)，默认为y: ")
                            if use_found.lower() != "n":
                                steps = len(image_paths)
                                wp.n = steps
                                print(f"相移步数已调整为 {steps}")
                            else:
                                # 如果找到的文件太多，截取前steps个
                                if len(image_paths) > steps:
                                    image_paths = image_paths[:steps]
                                    print(f"使用前 {steps} 个文件")
                                # 如果找到的文件太少，将使用测试图像补充
                                else:
                                    print(f"将使用测试图像补充缺失的文件")
                    elif len(image_paths) > 0:
                        print(f"在文件夹 {folder_path} 中找到 {len(image_paths)} 个图像文件，但至少需要3个")
                        print("将使用测试图像代替缺失的文件")
                    else:
                        print(f"在文件夹 {folder_path} 中未找到任何所需的图像文件")
                        
                        # 尝试查找任何图像文件
                        all_images = []
                        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                            all_images.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
                        
                        if all_images:
                            print(f"在文件夹中找到 {len(all_images)} 个图像文件:")
                            for i, img in enumerate(all_images[:8]):  # 最多显示8个
                                print(f"  {i+1}. {os.path.basename(img)}")
                            
                            use_found_images = input("是否使用这些图像? (y/n)，默认为y: ")
                            if use_found_images.lower() != "n":
                                # 使用找到的图像
                                image_paths = sorted(all_images)
                                if len(image_paths) >= 3:
                                    # 如果图像数量不等于指定的步数，调整步数
                                    if len(image_paths) != steps:
                                        print(f"警告: 找到 {len(image_paths)} 个图像文件，但指定的相移步数为 {steps}")
                                        use_found = input(f"是否使用这 {len(image_paths)} 个文件作为相移步数? (y/n)，默认为y: ")
                                        if use_found.lower() != "n":
                                            steps = len(image_paths)
                                            wp.n = steps
                                            print(f"相移步数已调整为 {steps}")
                                        else:
                                            # 如果找到的文件太多，截取前steps个
                                            if len(image_paths) > steps:
                                                image_paths = image_paths[:steps]
                                                print(f"使用前 {steps} 个文件")
                                    print(f"使用 {len(image_paths)} 个图像文件")
                                else:
                                    print(f"错误: 至少需要3个图像文件，但只找到 {len(image_paths)} 个")
                                    image_paths = None
                            else:
                                image_paths = None
                        else:
                            print("文件夹中没有找到任何图像文件，将使用测试图像")
                            image_paths = None
                else:
                    # 不是文件夹路径，尝试使用通配符匹配文件
                    matched_files = sorted(glob.glob(folder_or_pattern))
                    
                    if matched_files and len(matched_files) >= 3:
                        print(f"找到{len(matched_files)}个匹配的文件:")
                        for i, f in enumerate(matched_files[:8]):  # 最多显示8个
                            print(f"  {i+1}. {os.path.basename(f)}")
                        
                        # 如果找到的文件数量不等于指定的步数，询问是否调整步数
                        if len(matched_files) != steps:
                            print(f"警告: 找到 {len(matched_files)} 个图像文件，但指定的相移步数为 {steps}")
                            use_found = input(f"是否使用这 {len(matched_files)} 个文件作为相移步数? (y/n)，默认为y: ")
                            if use_found.lower() != "n":
                                steps = len(matched_files)
                                wp.n = steps
                                image_paths = matched_files
                                print(f"相移步数已调整为 {steps}")
                            else:
                                # 如果找到的文件太多，让用户选择或使用前steps个
                                if len(matched_files) > steps:
                                    select_option = input(f"找到超过{steps}个文件，是否手动选择文件? (y/n)，默认为n: ")
                                    if select_option.lower() == "y":
                                        # 手动选择文件
                                        print(f"请选择要使用的{steps}个文件，输入对应的编号(1-{len(matched_files)})，用空格分隔:")
                                        try:
                                            selections = input(f"选择 (例如: 1 2 ... {steps}): ").split()
                                            indices = [int(s) - 1 for s in selections]
                                            image_paths = [matched_files[i] for i in indices if 0 <= i < len(matched_files)]
                                            
                                            if len(image_paths) != steps:
                                                print(f"警告: 您选择了{len(image_paths)}个文件，但需要{steps}个。将使用可用文件，缺失的将用测试图像代替。")
                                        except Exception as e:
                                            print(f"选择文件时出错: {e}，将使用前{steps}个文件")
                                            image_paths = matched_files[:steps]
                                    else:
                                        # 使用前steps个文件
                                        image_paths = matched_files[:steps]
                                        print(f"使用前{steps}个文件: {[os.path.basename(p) for p in image_paths]}")
                                else:
                                    # 如果找到的文件太少，将使用测试图像补充
                                    image_paths = matched_files
                                    print(f"将使用测试图像补充缺失的文件")
                        else:
                            image_paths = matched_files
                    else:
                        print(f"警告: 未找到匹配的文件或文件数量不足 ({len(matched_files) if matched_files else 0})")
                        # 手动输入路径
                        image_paths = []
                        for i in range(steps):
                            phase = i * 360 / steps
                            path = input(f"请输入相移量为{phase:.1f}°的图像路径: ")
                            if path and os.path.exists(path):
                                image_paths.append(path)
                            else:
                                if path:
                                    print(f"警告: 文件不存在: {path}")
                                print("将使用测试图像代替")
            else:
                print("未提供路径，将使用测试图像")
                image_paths = None
        
        # 获取图像
        try:
            images = wp.getImageData(image_paths, direction, wp.standard_size, wp.size_method, n=steps)
        except Exception as e:
            print(f"获取图像时出错: {e}")
            print("将使用完全生成的测试图像")
            images = wp.getImageData(None, direction, wp.standard_size, wp.size_method, n=steps)
    
    # 检查是否成功获取了足够的图像
    valid_images = [img for img in images if img is not None]
    if len(valid_images) < 3:
        print(f"错误: 至少需要3幅相移图像，但只获取到{len(valid_images)}幅有效图像")
        return
    elif len(valid_images) < wp.n:
        print(f"警告: 需要{wp.n}幅相移图像，但只获取到{len(valid_images)}幅有效图像")
        print("将尝试使用可用图像计算包裹相位，结果可能不准确")
    
    # 计算包裹相位
    print(f"\n使用{len(valid_images)}步相移法计算包裹相位...")
    try:
        wrapped_phase = wp.computeWrappedphase(images)
        
        # 打印包裹相位的范围
        print(f"包裹相位范围: [{np.min(wrapped_phase)}, {np.max(wrapped_phase)}]")
        print(f"理论包裹相位范围: [0, {2*math.pi}]")
        
        # 保存结果
        prefix = "vertical_fringe_" if direction == "vertical" else "horizontal_fringe_"
        prefix += f"{len(valid_images)}step_"
        wp.save_wrapped_phase(wrapped_phase, args.output, prefix, direction=direction)
        
        # 如果需要可视化，显示包裹相位
        if args.visualize or not args.images:  # 如果是交互模式或指定了visualize参数
            title = f"{'垂直条纹' if direction == 'vertical' else '水平条纹'} - {len(valid_images)}步相移包裹相位"
            wp.visualize_wrapped_phase(wrapped_phase, title)
        
        print("\n包裹相位计算完成!")
    except Exception as e:
        print(f"计算包裹相位时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 