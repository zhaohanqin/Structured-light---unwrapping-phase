import numpy as np
import cv2 as cv
import math
import os

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
    
    # 过滤掉None值
    valid_images = [img for img in images if img is not None]
    if len(valid_images) == 0:
        return images
    
    # 如果未指定目标尺寸，计算所有图像的最小尺寸
    if target_size is None:
        min_height = min(img.shape[0] for img in valid_images)
        min_width = min(img.shape[1] for img in valid_images)
        target_size = (min_height, min_width)
    
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
    for img in images:
        if img is None:
            result.append(None)
            continue
        
        if method == "crop":
            # 裁剪到目标尺寸（从左上角开始）
            adjusted_img = img[:target_size[0], :target_size[1]]
        elif method == "resize":
            # 缩放到目标尺寸
            adjusted_img = cv.resize(img, (target_size[1], target_size[0]))
        else:
            raise ValueError(f"不支持的调整方法: {method}")
        
        result.append(adjusted_img)
    
    return result

# 整合 GrayCode 类
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
    
    def __init__(self, n: int = 3):
        """
        初始化格雷码生成器
        
        参数:
            n: 格雷码位数，默认为3位
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


# Phase Unwrapper 类 - 专注于相位解包裹
class PhaseUnwrapper:
    """
    相位解包裹器
    
    该类专注于相位解包裹功能，接收包裹相位图像和格雷码图像作为输入，
    对格雷码图像进行二值化处理，然后进行相位解包裹操作。
    """
    
    def __init__(self, n: int = 5, direction: str = "horizontal"):
        """
        初始化相位解包裹器
        
        参数:
            n (int): 格雷码位数，默认为5
            direction (str): 条纹方向，可选值为"horizontal"、"vertical"或"both"，默认为"horizontal"
        """
        self.n = n
        self.direction = direction
        self.standard_size = None  # 标准图像尺寸
        self.size_method = "crop"  # 默认使用裁剪方法
    
    def binarize_graycode_images(self, graycode_images, threshold_value=127):
        """
        对格雷码图像进行二值化处理
        
        参数:
            graycode_images: 格雷码图像列表
            threshold_value: 二值化阈值，默认为127
            
        返回:
            list: 二值化后的格雷码图像列表
        """
        binary_images = []
        
        for img in graycode_images:
            # 确保图像是灰度图
            if len(img.shape) > 2:
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # 进行二值化处理
            _, binary = cv.threshold(gray, threshold_value, 255, cv.THRESH_BINARY)
            
            # 将像素值从[0,255]归一化到[0,1]范围
            binary_scaled = binary / 255
            
            # 转换为uint8类型，此时像素值为0或1
            binary_images.append(binary_scaled.astype(np.uint8))
        
        return binary_images
    
    def unwrap_phase(self, wrapped_phase, graycode_images, adaptive_threshold=False, show_results=True, basename=""):
        """
        解包裹相位
        
        参数:
            wrapped_phase: 包裹相位图像，范围[0, 2π]
            graycode_images: 格雷码图像列表
            adaptive_threshold: 是否使用自适应阈值进行二值化，默认为False
            show_results: 是否显示结果，默认为True
            basename: 文件名前缀，用于区分不同图像的结果
            
        返回:
            numpy.ndarray: 解包裹相位图像
        """
        # 标准化图像尺寸
        all_images = [wrapped_phase] + graycode_images
        normalized_images = normalize_image_size(all_images, self.standard_size, self.size_method)
        
        wrapped_phase = normalized_images[0]
        graycode_images = normalized_images[1:]
        
        # 确保有足够数量的格雷码图像
        if len(graycode_images) < self.n:
            raise ValueError(f"需要至少 {self.n} 张格雷码图像，但只提供了 {len(graycode_images)} 张")
        
        # 二值化格雷码图像
        if adaptive_threshold:
            # 使用自适应阈值
            binary_images = []
            for img in graycode_images:
                if len(img.shape) > 2:
                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                else:
                    gray = img
                
                binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv.THRESH_BINARY, 11, 2)
                binary_scaled = binary / 255
                binary_images.append(binary_scaled.astype(np.uint8))
        else:
            # 使用固定阈值
            binary_images = self.binarize_graycode_images(graycode_images)
        
        # 获取k1和k2矩阵
        k1, k2 = self.get_k1_k2_from_binary(binary_images)
        
        # 检查包裹相位和k1、k2的尺寸是否一致
        print(f"包裹相位尺寸: {wrapped_phase.shape}")
        print(f"k1尺寸: {k1.shape}")
        print(f"k2尺寸: {k2.shape}")
        
        # 确保尺寸一致
        if wrapped_phase.shape != k1.shape or wrapped_phase.shape != k2.shape:
            # 调整到相同尺寸
            min_rows = min(wrapped_phase.shape[0], k1.shape[0], k2.shape[0])
            min_cols = min(wrapped_phase.shape[1], k1.shape[1], k2.shape[1])
            
            wrapped_phase = wrapped_phase[:min_rows, :min_cols]
            k1 = k1[:min_rows, :min_cols]
            k2 = k2[:min_rows, :min_cols]
            
            print(f"调整后尺寸: {min_rows}x{min_cols}")
        
        # 使用相位连续性约束的解包裹算法
        print("使用相位连续性约束的解包裹算法...")
        unwrapped_phase = self.phase_unwrapping_with_continuity(wrapped_phase, k1, k2)
        
        # 限制解包裹相位的范围
        max_phase = 32 * math.pi  # 5位格雷码的理论最大值
        unwrapped_phase = np.clip(unwrapped_phase, 0, max_phase)
        
        # 打印解包裹相位的范围
        print(f"解包裹相位的范围: [{np.min(unwrapped_phase)}, {np.max(unwrapped_phase)}]")
        print(f"理论范围: [0, {32*math.pi}]")
        
        # 对解包裹相位进行平滑处理
        print("对解包裹相位进行平滑处理...")
        smoothed_phase = self.smooth_unwrapped_phase(unwrapped_phase)
        
        if show_results:
            # 保存和显示结果
            self.save_and_display_results(unwrapped_phase, smoothed_phase, self.direction, basename)
        
        return smoothed_phase
    
    def get_k1_k2_from_binary(self, binary_graycodes):
        """
        从二值化格雷码图像获取k1和k2矩阵
        
        参数:
            binary_graycodes: 二值化后的格雷码图像列表
            
        返回:
            tuple: (k1矩阵, k2矩阵)
        """
        # 确保有足够数量的格雷码图像
        if len(binary_graycodes) < 5:
            raise ValueError(f"需要至少5张二值化格雷码图像，但只提供了{len(binary_graycodes)}张")
        
        rows, cols = binary_graycodes[0].shape
        
        # 初始化k1和k2矩阵
        k1 = np.zeros((rows, cols), np.uint8)  # 4位格雷码解码结果
        k2 = np.zeros((rows, cols), np.uint8)  # 5位格雷码解码结果
        
        # 创建格雷码映射对象
        g_k1 = GrayCode(4)  # 4位格雷码映射器
        g_k2 = GrayCode(5)  # 5位格雷码映射器
        
        # 逐像素解码格雷码
        for a in range(rows):
            for b in range(cols):
                code1 = ""
                
                # 组合前4位格雷码，用于计算k1
                code_k1 = code1 + str(binary_graycodes[0][a,b]) + str(binary_graycodes[1][a,b]) + str(binary_graycodes[2][a,b]) + str(binary_graycodes[3][a,b])
                
                # 组合前5位格雷码，用于计算k2
                code_k2 = code1 + str(binary_graycodes[0][a,b]) + str(binary_graycodes[1][a,b]) + str(binary_graycodes[2][a,b]) + str(binary_graycodes[3][a,b]) + str(binary_graycodes[4][a,b])
                
                try:
                    # 查询格雷码映射表，获得对应的十进制值
                    k1[a,b] = g_k1.code2k[code_k1]  # 4位格雷码对应的十进制数
                    k2[a,b] = g_k2.code2k[code_k2]  # 5位格雷码对应的十进制数
                except KeyError as e:
                    # 对于解码错误的情况，使用相邻像素的值或默认值
                    if a > 0:
                        k1[a,b] = k1[a-1,b]
                        k2[a,b] = k2[a-1,b]
                    elif b > 0:
                        k1[a,b] = k1[a,b-1]
                        k2[a,b] = k2[a,b-1]
                    else:
                        k1[a,b] = 0
                        k2[a,b] = 0
        
        return k1, k2
    
    def phase_unwrapping_with_continuity(self, wrapped_pha, k1, k2):
        """
        添加相位连续性约束的解包裹算法
        
        该方法结合包裹相位和k值矩阵，同时考虑相位连续性约束，
        计算解包裹相位。相位连续性约束可以有效减少相位跳变，
        提高解包裹结果的质量。
        
        算法流程：
        1. 处理第一个像素点，确定其绝对相位
        2. 逐像素处理剩余点，考虑相邻像素的相位值
        3. 对于每个像素，选择与相邻像素相位最接近的解包裹值
        
        参数:
            wrapped_pha: 包裹相位矩阵，范围[0, 2π]
            k1: 基于4位格雷码的周期编号矩阵
            k2: 基于5位格雷码的周期编号矩阵
            
        返回:
            numpy.ndarray: 解包裹相位矩阵
        """
        rows, cols = wrapped_pha.shape
        unwrapped_pha = np.zeros_like(wrapped_pha, dtype=np.float32)
        
        # 先处理第一个像素
        if wrapped_pha[0, 0] <= math.pi/2:
            # 当包裹相位在[0, π/2]范围内，使用k2值确定周期
            unwrapped_pha[0, 0] = wrapped_pha[0, 0] + k2[0, 0]*2*math.pi
        elif wrapped_pha[0, 0] < 3*math.pi/2:
            # 当包裹相位在(π/2, 3π/2)范围内，使用k1值确定周期
            unwrapped_pha[0, 0] = wrapped_pha[0, 0] + k1[0, 0]*2*math.pi
        else:
            # 当包裹相位在[3π/2, 2π)范围内，使用k2值确定周期
            unwrapped_pha[0, 0] = wrapped_pha[0, 0] + k2[0, 0]*2*math.pi
        
        # 逐行处理，确保行内相位连续
        for r in range(rows):
            for c in range(cols):
                if r == 0 and c == 0:
                    continue  # 第一个像素已处理
                    
                # 获取相邻像素的相位值
                neighbors = []
                if c > 0:
                    neighbors.append(unwrapped_pha[r, c-1])  # 左邻居
                if r > 0:
                    neighbors.append(unwrapped_pha[r-1, c])  # 上邻居
                    
                if not neighbors:
                    # 没有有效邻居，使用原始算法
                    if wrapped_pha[r, c] <= math.pi/2:
                        unwrapped_pha[r, c] = wrapped_pha[r, c] + k2[r, c]*2*math.pi
                    elif wrapped_pha[r, c] < 3*math.pi/2:
                        unwrapped_pha[r, c] = wrapped_pha[r, c] + k1[r, c]*2*math.pi
                    else:
                        unwrapped_pha[r, c] = wrapped_pha[r, c] + k2[r, c]*2*math.pi
                else:
                    # 有邻居，考虑相位连续性
                    avg_neighbor = np.mean(neighbors)
                    
                    # 计算不同周期下的相位值
                    candidates = []
                    # 考虑邻近的3个周期
                    for k in range(max(0, int(avg_neighbor/(2*math.pi))-1), int(avg_neighbor/(2*math.pi))+2):
                        candidates.append(wrapped_pha[r, c] + k*2*math.pi)
                    
                    # 选择与邻居平均值最接近的相位值
                    unwrapped_pha[r, c] = min(candidates, key=lambda x: abs(x - avg_neighbor))
        
        return unwrapped_pha
    
    def smooth_unwrapped_phase(self, unwrapped_pha, kernel_size=5):
        """
        对解包裹相位进行平滑处理
        
        该方法结合中值滤波和高斯滤波，对解包裹相位进行平滑处理，
        以减少噪声和相位跳变。
        
        参数:
            unwrapped_pha: 解包裹相位矩阵
            kernel_size: 滤波核大小，默认为5
            
        返回:
            numpy.ndarray: 平滑后的相位矩阵
        """
        # 将相位矩阵转换为float32类型，确保滤波操作的精度
        phase_float32 = unwrapped_pha.astype(np.float32)
        
        # 使用中值滤波去除离群值（相位跳变点）
        median_filtered = cv.medianBlur(phase_float32, kernel_size)
        
        # 使用高斯滤波进一步平滑
        smoothed = cv.GaussianBlur(median_filtered, (kernel_size, kernel_size), 0)
        
        return smoothed
    
    def save_and_display_results(self, unwrapped_phase, smoothed_phase, direction="horizontal", basename=""):
        """
        保存和显示解包裹相位结果
        
        参数:
            unwrapped_phase: 原始解包裹相位
            smoothed_phase: 平滑后的解包裹相位
            direction: 解包裹方向，可选值为"horizontal"或"vertical"，默认为"horizontal"
            basename: 文件名前缀，用于区分不同图像的结果
        """
        # 创建results目录（如果不存在）
        if not os.path.exists("results"):
            os.makedirs("results")
        
        # 构建文件名前缀
        if basename:
            prefix = f"results/{basename}_{direction}"
        else:
            prefix = f"results/{direction}"

        # 将相位值缩放到[0,255]范围
        unwrapped_scaled = (unwrapped_phase * 255 / np.max(unwrapped_phase)).astype(np.uint8)
        smoothed_scaled = (smoothed_phase * 255 / np.max(smoothed_phase)).astype(np.uint8)
        
        # 应用伪彩色映射以增强可视化效果
        unwrapped_color = cv.applyColorMap(unwrapped_scaled, cv.COLORMAP_JET)
        smoothed_color = cv.applyColorMap(smoothed_scaled, cv.COLORMAP_JET)
        
        # 保存结果
        cv.imwrite(f"{prefix}_unwrapped_phase_original.png", unwrapped_color)
        cv.imwrite(f"{prefix}_unwrapped_phase_smoothed.png", smoothed_color)
        
        # 保存为16位PNG，以保留更多细节
        unwrapped_norm = unwrapped_phase / np.max(unwrapped_phase)
        smoothed_norm = smoothed_phase / np.max(smoothed_phase)
        cv.imwrite(f"{prefix}_unwrapped_phase_height.png", (unwrapped_norm * 65535).astype(np.uint16))
        cv.imwrite(f"{prefix}_smoothed_phase_height.png", (smoothed_norm * 65535).astype(np.uint16))
        
        # 显示结果
        window_name_orig = f"原始解包裹相位 - {basename}" if basename else "原始解包裹相位"
        window_name_smooth = f"平滑后的解包裹相位 - {basename}" if basename else "平滑后的解包裹相位"
        cv.imshow(window_name_orig, unwrapped_color)
        cv.imshow(window_name_smooth, smoothed_color)
        
        # 定义鼠标回调函数，显示点击位置的相位值
        def mouse_callback(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                # 获取窗口名称（用于区分是哪个窗口被点击）
                window_name = param
                
                # 获取点击位置的相位值
                if window_name == window_name_orig:
                    phase_value = unwrapped_phase[y, x]
                else:
                    phase_value = smoothed_phase[y, x]
                
                # 计算相位值对应的周期数
                period = phase_value / (2 * math.pi)
                
                # 在控制台显示信息
                print(f"点击位置 ({x}, {y}):")
                print(f"  相位值: {phase_value:.6f} rad")
                print(f"  周期数: {period:.6f}")
                
                # 在图像上显示相位值
                if window_name == window_name_orig:
                    display_img = unwrapped_color.copy()
                else:
                    display_img = smoothed_color.copy()
                
                # 绘制十字标记
                cv.drawMarker(display_img, (x, y), (0, 255, 255), cv.MARKER_CROSS, 20, 2)
                
                # 添加文本
                text = f"Phase: {phase_value:.2f} rad"
                cv.putText(display_img, text, (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # 更新显示
                cv.imshow(window_name, display_img)
        
        # 设置鼠标回调
        cv.setMouseCallback(window_name_orig, mouse_callback, window_name_orig)
        cv.setMouseCallback(window_name_smooth, mouse_callback, window_name_smooth)
        
        # 显示交互提示
        print("\n交互提示:")
        print("- 鼠标点击图像可显示该点的相位值")
        print("- 按's'键保存图像")
        print("- 按'q'键或ESC键退出")
        
        # 等待用户按键
        while True:
            key = cv.waitKey(0)
            if key == ord("s"):
                # 按's'键保存图像
                cv.imwrite(f"{prefix}_unwrapped_phase_user.png", unwrapped_color)
                cv.imwrite(f"{prefix}_smoothed_phase_user.png", smoothed_color)
                print(f"已保存图像到results目录")
            elif key == ord("q") or key == 27:  # 'q'键或ESC键
                break
        
        # 关闭所有图像窗口
        cv.destroyAllWindows()


def generate_combined_phase_image(h_unwrapped, v_unwrapped, output_path=None):
    """
    生成水平和垂直方向相位组合图，并提供交互式显示
    
    参数:
        h_unwrapped: 水平方向解包裹相位
        v_unwrapped: 垂直方向解包裹相位
        output_path: 输出路径 (可选)
    
    返回:
        combined_rgb: 组合的RGB图像
    """
    if h_unwrapped is None or v_unwrapped is None:
        print("需要水平和垂直方向的相位数据才能生成组合图")
        return None
        
    # 确保两个相位图具有相同的大小
    if h_unwrapped.shape != v_unwrapped.shape:
        print(f"水平和垂直方向相位图尺寸不一致 (H:{h_unwrapped.shape}, V:{v_unwrapped.shape})，正在尝试裁剪为相同尺寸。")
        images = normalize_image_size([h_unwrapped, v_unwrapped], method="crop")
        h_unwrapped, v_unwrapped = images[0], images[1]
        print(f"调整后统一尺寸为: {h_unwrapped.shape}")
        
    height, width = h_unwrapped.shape
    
    # 归一化两个相位图用于显示
    h_min, h_max = np.min(h_unwrapped), np.max(h_unwrapped)
    v_min, v_max = np.min(v_unwrapped), np.max(v_unwrapped)
    
    h_norm = (h_unwrapped - h_min) / (h_max - h_min + 1e-9)
    v_norm = (v_unwrapped - v_min) / (v_max - v_min + 1e-9)
    
    # 组合两个方向的相位图得到伪彩色图像
    combined_rgb = np.zeros((height, width, 3), dtype=np.float32)
    combined_rgb[:,:,0] = h_norm  # 红色通道为水平方向
    combined_rgb[:,:,1] = v_norm  # 绿色通道为垂直方向
    combined_rgb[:,:,2] = (h_norm + v_norm) / 2  # 蓝色通道为两者平均
    
    # 如果指定了输出路径，保存并进行交互式显示
    if output_path:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 转换为OpenCV可用的BGR格式并保存
        combined_bgr_uint8 = cv.cvtColor((combined_rgb * 255).astype(np.uint8), cv.COLOR_RGB2BGR)
        cv.imwrite(output_path, combined_bgr_uint8)
        print(f"已保存组合相位图到: {output_path}")

        # --- 交互式显示 ---
        window_name = "Combined Phase Map (Interactive)"

        # 定义鼠标回调函数
        def mouse_callback_combined(event, x, y, flags, params):
            if event == cv.EVENT_LBUTTONDOWN:
                h_phase_map, v_phase_map, base_img = params
                
                # 获取点击位置的相位值
                h_phase_val = h_phase_map[y, x]
                v_phase_val = v_phase_map[y, x]
                
                print(f"点击位置 ({x}, {y}):")
                print(f"  水平相位值: {h_phase_val:.4f} rad")
                print(f"  垂直相位值: {v_phase_val:.4f} rad")
                
                display_img = base_img.copy()
                
                cv.drawMarker(display_img, (x, y), (0, 255, 255), cv.MARKER_CROSS, 20, 2)
                
                text_h = f"H-Phase: {h_phase_val:.2f}"
                text_v = f"V-Phase: {v_phase_val:.2f}"
                cv.putText(display_img, text_h, (x + 15, y - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv.putText(display_img, text_v, (x + 15, y + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                cv.imshow(window_name, display_img)

        cv.imshow(window_name, combined_bgr_uint8)
        cv.setMouseCallback(window_name, mouse_callback_combined, (h_unwrapped, v_unwrapped, combined_bgr_uint8))

        print("\n--- 组合图像交互 ---")
        print("- 鼠标点击图像可显示该点的水平和垂直相位值")
        print("- 按's'键保存原始组合图像的副本")
        print("- 按'q'键或ESC键退出")

        while True:
            key = cv.waitKey(0) & 0xFF
            if key == ord("s"):
                user_save_path = output_path.replace(".png", "_user_saved.png")
                cv.imwrite(user_save_path, combined_bgr_uint8)
                print(f"已保存图像到: {user_save_path}")
            elif key == ord("q") or key == 27:
                break
        
        cv.destroyAllWindows()

    return combined_rgb


# 示例使用
if __name__ == "__main__":
    print("相位解包裹程序 (水平与垂直方向)")
    print("==============================================")

    # --- 定义一个可重用的处理函数 ---
    def process_unwrapping(direction, wrapped_phase_input, graycode_folder, n_graycodes, use_adaptive_threshold, show_results=True):
        """
        封装的解包裹处理流程
        
        参数:
            direction (str): 解包裹方向 ("horizontal" 或 "vertical")
            wrapped_phase_input (str): 包裹相位图像文件或文件夹路径
            graycode_folder (str): 格雷码图像文件夹路径
            n_graycodes (int): 需要的格雷码图像数量
            use_adaptive_threshold (bool): 是否使用自适应阈值
            show_results (bool): 是否显示单个解包裹结果
        
        返回:
            list: 解包裹后的相位图列表
            list: 对应的文件名列表 (无扩展名)
        """
        print(f"\n{'='*20}")
        if direction == "horizontal":
            print("  开始进行 [水平方向] 解包裹  ")
            print("  (使用垂直条纹格雷码)  ")
        else:
            print("  开始进行 [垂直方向] 解包裹  ")
            print("  (使用水平条纹格雷码)  ")
        print(f"{'='*20}")

        # 检查路径是否存在
        if not os.path.exists(wrapped_phase_input) or not os.path.isdir(graycode_folder):
            print(f"错误: 提供的路径之一无效，跳过 {direction} 方向解包裹。")
            if not os.path.exists(wrapped_phase_input):
                print(f"  - 包裹相位路径不存在: {wrapped_phase_input}")
            if not os.path.isdir(graycode_folder):
                print(f"  - 格雷码文件夹不存在: {graycode_folder}")
            return [], []

        # 创建解包裹器
        unwrapper = PhaseUnwrapper(n=n_graycodes, direction=direction)
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

        # 加载格雷码图像
        print(f"\n从文件夹 {graycode_folder} 读取格雷码图像...")
        graycode_images = []
        graycode_files = sorted([f for f in os.listdir(graycode_folder) if f.lower().endswith(image_extensions)])
        
        if len(graycode_files) < unwrapper.n:
            print(f"错误: 在文件夹 {graycode_folder} 中没有找到足够的格雷码图像 (需要至少{unwrapper.n}张，找到了{len(graycode_files)}张)")
            return [], []

        for fname in graycode_files[:unwrapper.n]:
            path = os.path.join(graycode_folder, fname)
            img = cv.imread(path, -1)
            if img is None:
                print(f"错误: 无法读取格雷码图像 {path}")
                return [], []
            graycode_images.append(img)
        print(f"成功读取 {len(graycode_images)} 张格雷码图像。")

        # 加载并处理每个包裹相位图像
        wrapped_phase_paths = []
        if os.path.isdir(wrapped_phase_input):
            print(f"\n从文件夹 {wrapped_phase_input} 读取包裹相位图像...")
            print("注意: 将仅查找文件名中包含 'wrapped_phase_equalized' 的图像文件。")
            wrapped_phase_files = sorted([
                f for f in os.listdir(wrapped_phase_input) 
                if "wrapped_phase_equalized" in f.lower() and f.lower().endswith(image_extensions)
            ])
            if not wrapped_phase_files:
                print(f"错误: 在文件夹 {wrapped_phase_input} 中没有找到包含 'wrapped_phase_equalized' 的包裹相位图像。")
                return [], []
            for f in wrapped_phase_files:
                wrapped_phase_paths.append(os.path.join(wrapped_phase_input, f))
        elif os.path.isfile(wrapped_phase_input):
            print(f"\n读取包裹相位图像文件 {wrapped_phase_input}...")
            if wrapped_phase_input.lower().endswith(image_extensions):
                wrapped_phase_paths.append(wrapped_phase_input)
            else:
                print(f"错误:提供的文件 {wrapped_phase_input} 不是支持的图像格式。")
                return [], []

        if not wrapped_phase_paths:
            print(f"错误: 在路径 {wrapped_phase_input} 中没有找到有效的包裹相位图像。")
            return [], []

        all_unwrapped_phases = []
        all_basenames = []
        for wrapped_phase_path in wrapped_phase_paths:
            wrapped_phase_file = os.path.basename(wrapped_phase_path)
            print(f"\n--- 正在处理: {wrapped_phase_file} ({direction}) ---")
            
            wrapped_phase = cv.imread(wrapped_phase_path, -1)
            if wrapped_phase is None:
                print(f"警告: 无法读取包裹相位图像 {wrapped_phase_path}，已跳过。")
                continue
        
            # 转换为float32类型并确保范围在[0, 2π]
            if wrapped_phase.dtype != np.float32:
                if wrapped_phase.dtype == np.uint8:
                    wrapped_phase = wrapped_phase.astype(np.float32) * (2 * math.pi / 255)
                elif wrapped_phase.dtype == np.uint16:
                    wrapped_phase = wrapped_phase.astype(np.float32) * (2 * math.pi / 65535)
                else:
                    wrapped_phase = wrapped_phase.astype(np.float32)
            
            basename = os.path.splitext(wrapped_phase_file)[0]

            # 执行解包裹
            smoothed_phase = unwrapper.unwrap_phase(
                wrapped_phase,
                graycode_images,
                adaptive_threshold=use_adaptive_threshold,
                show_results=show_results,
                basename=basename
            )
            all_unwrapped_phases.append(smoothed_phase)
            all_basenames.append(basename)
            print(f"--- 完成处理: {wrapped_phase_file} ({direction}) ---")
        
        return all_unwrapped_phases, all_basenames

    # --- 获取用户输入 ---
    print("\n请选择要执行的解包裹模式:")
    print("  1. 仅水平方向解包裹")
    print("  2. 仅垂直方向解包裹")
    print("  3. 同时进行水平和垂直方向解包裹")
    choice = input("请输入您的选择 (1/2/3)，输入其他则退出: ")

    # --- 通用设置 ---
    if choice in ['1', '2', '3']:
        N_GRAYCODES = 5  # 默认使用5位格雷码
        adaptive_choice = input("\n是否对所有图像使用自适应阈值进行二值化? (y/n)，默认为n: ")
        adaptive_threshold = adaptive_choice.lower() == "y"

        h_unwrapped_results, v_unwrapped_results = [], []
        h_basenames, v_basenames = [], []

        # --- 执行解包裹流程 ---
        if choice == '1' or choice == '3':
            # 1. 水平解包裹
            print("\n--- 设置水平方向解包裹的路径 ---")
            h_wrapped_phase_input = input("请输入[水平方向]包裹相位图像的文件或文件夹路径: ")
            v_graycode_folder = input("请输入[垂直条纹]格雷码图像所在的文件夹路径: ")
            h_unwrapped_results, h_basenames = process_unwrapping(
                direction="horizontal",
                wrapped_phase_input=h_wrapped_phase_input,
                graycode_folder=v_graycode_folder,
                n_graycodes=N_GRAYCODES,
                use_adaptive_threshold=adaptive_threshold,
                show_results=(choice == '1')
            )

        if choice == '2' or choice == '3':
            # 2. 垂直解包裹
            print("\n--- 设置垂直方向解包裹的路径 ---")
            v_wrapped_phase_input = input("请输入[垂直方向]包裹相位图像的文件或文件夹路径: ")
            h_graycode_folder = input("请输入[水平条纹]格雷码图像所在的文件夹路径: ")
            v_unwrapped_results, v_basenames = process_unwrapping(
                direction="vertical",
                wrapped_phase_input=v_wrapped_phase_input,
                graycode_folder=h_graycode_folder,
                n_graycodes=N_GRAYCODES,
                use_adaptive_threshold=adaptive_threshold,
                show_results=(choice == '2')
            )
        
        # 如果选择3，则合并结果
        if choice == '3' and h_unwrapped_results and v_unwrapped_results:
            print("\n--- 正在合并水平和垂直方向的相位图 ---")
            
            # 检查两个方向的结果数量是否一致
            if len(h_unwrapped_results) != len(v_unwrapped_results):
                print(f"警告: 水平和垂直方向的结果数量不一致 (H:{len(h_unwrapped_results)}, V:{len(v_unwrapped_results)})，无法合并。")
                print("请确保两个方向的输入包含相同数量的图像。")
            else:
                # 按顺序配对结果进行合并
                for i, h_result in enumerate(h_unwrapped_results):
                    v_result = v_unwrapped_results[i]
                    h_basename = h_basenames[i]
                    v_basename = v_basenames[i]

                    # 创建一个通用的输出文件名，或者基于其中一个
                    # 这里我们使用水平方向图像的基名
                    output_path = os.path.join("results", f"{h_basename}_combined_phase.png")
                    print(f"\n正在为 '{h_basename}' 和 '{v_basename}' 生成组合相位图...")
                    
                    generate_combined_phase_image(
                        h_unwrapped=h_result,
                        v_unwrapped=v_result,
                        output_path=output_path
                    )

        print("\n所有图像处理完成!")
    else:
        print("无效选择或选择退出，程序结束。") 