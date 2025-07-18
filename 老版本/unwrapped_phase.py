import numpy as np
import cv2 as cv
import math
import os
from phase_unwrapper import PhaseUnwrapper

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

# 整合 GrayCode 类 (原 generate_graycode_map.py)
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

    def toPattern(self, idx: int, cols: int = 1920, rows: int = 1080):
        '''
        生成格雷码光栅图
        
        将格雷码转换为投影用的光栅图案
        
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


# 整合 WrappedPhase 类 (原 wrapped_phase_algorithm.py)
class WrappedPhase():
    """
    包裹相位计算类
    
    该类实现了四步相移法计算包裹相位的核心算法。
    四步相移法通过投影四幅具有不同相移量的正弦条纹图案，
    然后根据采集到的四幅图像计算每个像素点的包裹相位值。
    """
    
    def __init__(self, n: int = 4):
        """
        初始化包裹相位计算器
        
        
        参数:
            n (int): 相移步数，默认为4（四步相移法）
        """
        self.n = n
        # 保存标准尺寸，用于所有图像的标准化
        self.standard_size = None
        self.size_method = "crop"  # 默认使用裁剪方法
    
    @staticmethod
    def getImageData(m: int = 4, direction: str = "horizontal", standard_size=None, size_method="crop"):
        """
        获取相机拍摄的n幅相移图像
        
        该方法从指定路径读取四步相移法采集的图像序列。
        如果找不到图像文件，会生成测试用的相移图像。
        
        参数:
            m (int): 相移图像数量，默认为4
            direction (str): 条纹方向，可选值为"horizontal"或"vertical"，默认为"horizontal"
            standard_size: 标准图像尺寸(height, width)，默认为None（自动计算）
            size_method: 尺寸调整方法，可选值为"crop"(裁剪)或"resize"(缩放)
            
        返回:
            list: 包含m幅相移图像的列表，所有图像具有相同的尺寸
        """
        I = []
        for i in range(m):
            # 根据方向选择不同的图像文件
            if direction == "horizontal":
                # 水平方向使用I1-I4
                filename = r"fringe_patterns\I" + str(i+1) + ".png"
            else:
                # 垂直方向使用I5-I8
                filename = r"fringe_patterns\I" + str(i+5) + ".png"
            
            try:
                # 尝试直接读取图像
                img = cv.imread(filename, -1)
                if img is None:
                    # 如果直接读取失败，尝试使用二进制方式读取
                    img_file = np.fromfile(filename, dtype=np.uint8)
                    img = cv.imdecode(img_file, -1)
                
                if img is None:
                    raise FileNotFoundError(f"无法读取图像: {filename}")
            except Exception as e:
                print(f"读取图像 {filename} 失败: {e}")
                print("使用随机生成的测试图像代替...")
                # 生成随机图像用于测试（如果无法读取原始图像）
                height, width = 480, 640  # 默认测试图像大小
                # 创建带有随机噪声的正弦图案
                x = np.arange(0, width)
                y = np.arange(0, height).reshape(height, 1)
                phase = 2 * math.pi * i / m  # 相移量
                freq = 10  # 频率
                
                if direction == "horizontal":
                    # 水平条纹
                    img = np.sin(2 * math.pi * freq * x / width + phase) * 127 + 128
                else:
                    # 垂直条纹
                    img = np.sin(2 * math.pi * freq * y / height + phase) * 127 + 128
                
                img = img.astype(np.uint8)
            
            I.append(img)
        
        # 标准化所有图像的尺寸
        I = normalize_image_size(I, standard_size, size_method)
        
        # 打印图像尺寸信息
        if len(I) > 0 and I[0] is not None:
            print(f"相移图像尺寸: {I[0].shape[0]}x{I[0].shape[1]} ({direction}方向)")
        
        return I

    def computeWrappedphase(self, I):
        """
        计算包裹相位
        
        该方法实现了四步相移法的核心算法，通过四幅相移图像计算包裹相位。
        
        参数:
            I: 四幅相移图像列表 [I0, I1, I2, I3]，其中：
               I0: 相移量为0的图像
               I1: 相移量为π/2的图像
               I2: 相移量为π的图像
               I3: 相移量为3π/2的图像
        
        返回:
            numpy.ndarray: 包裹相位矩阵，范围[0, 2π]
        """
        # 将四幅图像转换为浮点数类型，提高计算精度
        i0 = I[0].astype(np.float32)  # 相移量为0的图像
        i1 = I[1].astype(np.float32)  # 相移量为π/2的图像
        i2 = I[2].astype(np.float32)  # 相移量为π的图像
        i3 = I[3].astype(np.float32)  # 相移量为3π/2的图像
        
        # 自动获取图像尺寸
        height, width = i0.shape
        
        # 初始化包裹相位矩阵
        pha = np.zeros((height, width), np.float32)
        
        # 逐像素计算包裹相位
        for a in range(height):
            for b in range(width):
                # 处理边界条件：分母为零的情况
                if i0[a,b] == i2[a,b] and i3[a,b] < i1[a,b]:
                    # 当I0=I2且I3<I1时，相位为3π/2
                    pha[a,b] = 3*math.pi/2
                elif i0[a,b] == i2[a,b] and i3[a,b] > i1[a,b]:
                    # 当I0=I2且I3>I1时，相位为π/2
                    pha[a,b] = math.pi/2
                elif i3[a, b] == i1[a, b] and i0[a, b] < i2[a, b]:
                    # 当I3=I1且I0<I2时，相位为π
                    pha[a, b] = math.pi
                elif i3[a, b] == i1[a, b] and i0[a, b] > i2[a, b]:
                    # 当I3=I1且I0>I2时，相位为0
                    pha[a, b] = 0
                # 处理四个象限的正常情况
                elif i0[a, b] > i2[a, b] and i1[a,b] < i3[a,b]:
                    # 第一象限：φ = arctan((I3-I1)/(I0-I2))
                    pha[a,b] = math.atan((i3[a,b] - i1[a, b])/ (i0[a, b] - i2[a, b]))
                elif i0[a, b] < i2[a, b] and i1[a,b] < i3[a,b]:
                    # 第二象限：φ = π - arctan((I3-I1)/(I2-I0))
                    pha[a,b] = math.pi-math.atan((i3[a,b] - i1[a, b])/ (i2[a, b] - i0[a, b]))
                elif i0[a, b] < i2[a, b] and i1[a,b] > i3[a,b]:
                    # 第三象限：φ = π + arctan((I3-I1)/(I0-I2))
                    pha[a,b] = math.pi + math.atan((i3[a,b] - i1[a, b])/ (i0[a, b] - i2[a, b]))
                elif i0[a, b] > i2[a, b] and i1[a,b] > i3[a,b]:
                    # 第四象限：φ = 2π - arctan((I1-I3)/(I0-I2))
                    pha[a,b] = 2*math.pi - math.atan((i1[a,b] - i3[a, b])/ (i0[a, b] - i2[a, b]))
        
        return pha


# 整合 Binariization 类 (原 graycode_binarization.py)
class Binariization():
    """
    格雷码二值化处理类
    
    该类用于对格雷码图像进行二值化处理，将灰度图像转换为二值图像，
    便于后续的格雷码解码。
    """

    def __init__(self, n:int=5):
        """
        初始化二值化处理器
        
        参数:
            n (int): 格雷码位数，默认为5
        """
        self.n = n
        self.standard_size = None  # 标准图像尺寸
        self.size_method = "crop"  # 默认使用裁剪方法

    def get_threshold(self, m:int = 4, direction:str = "horizontal", standard_size=None, size_method="crop"):
        '''
        利用四幅相移图计算阈值
        
        该方法通过计算四幅相移图像的平均值作为二值化阈值，
        这种方法可以自适应地处理不同光照条件下的图像。
        
        参数:
            m (int): 相移图像数量，默认为4
            direction (str): 条纹方向，可选值为"horizontal"或"vertical"，默认为"horizontal"
            standard_size: 标准图像尺寸(height, width)，默认为None（自动计算）
            size_method: 尺寸调整方法，可选值为"crop"(裁剪)或"resize"(缩放)
            
        返回:
            numpy.ndarray: 二值化阈值矩阵
        '''
        wp = WrappedPhase()
        # 更新标准尺寸
        self.standard_size = standard_size
        self.size_method = size_method
        
        # 获取标准化后的相移图像
        I = wp.getImageData(m, direction, standard_size, size_method)
        i = []
        for k in range(m):
            i.append(I[k].astype(np.float32))
        I_th = np.rint((i[0]+i[1]+i[2]+i[3])/m)  # np.rint()四舍五入取整
        TH = I_th.astype(np.uint8)
        
        # 保存标准尺寸供后续使用
        if len(I) > 0 and I[0] is not None:
            self.standard_size = I[0].shape[:2]
            
        return TH

    def get_GC_images(self, standard_size=None, size_method=None):
        '''
        读取格雷码图片
        
        该方法从指定路径读取格雷码图像。如果找不到图像文件，
        会生成测试用的格雷码图像。
        
        参数:
            standard_size: 标准图像尺寸，如果为None则使用之前计算的尺寸
            size_method: 尺寸调整方法，如果为None则使用之前设置的方法
            
        返回:
            list: 格雷码图像列表，所有图像具有相同的尺寸
        '''
        # 使用已保存的标准尺寸和方法（如果未指定）
        if standard_size is None:
            standard_size = self.standard_size
        if size_method is None:
            size_method = self.size_method
            
        J = []
        for i in range(5):
            try:
                filename = r"gray_patterns\gray_bit_" + str(i) + ".png"
                # 尝试直接读取图像
                img = cv.imread(filename, -1)
                if img is None:
                    # 如果直接读取失败，尝试使用二进制方式读取
                    file_img = np.fromfile(filename, dtype=np.uint8)
                    img = cv.imdecode(file_img, -1)
            except Exception as e:
                print(f"读取格雷码图像 {filename} 失败: {e}")
                print("使用随机生成的测试格雷码图像代替...")
                # 生成一个简单的测试格雷码图像
                g = GrayCode(5)
                img = g.toPattern(i, 640, 480)
            
            J.append(img)
            
        # 标准化所有格雷码图像的尺寸
        J = normalize_image_size(J, standard_size, size_method)
        
        # 打印图像尺寸信息
        if len(J) > 0 and J[0] is not None:
            print(f"格雷码图像尺寸: {J[0].shape[0]}x{J[0].shape[1]}")
            
            # 如果尺寸与之前的标准尺寸不同，更新标准尺寸
            if self.standard_size is not None and (J[0].shape[0] != self.standard_size[0] or J[0].shape[1] != self.standard_size[1]):
                print(f"警告: 格雷码图像尺寸({J[0].shape[0]}x{J[0].shape[1]})与相移图像尺寸({self.standard_size[0]}x{self.standard_size[1]})不一致")
                print(f"已将格雷码图像调整为: {J[0].shape[0]}x{J[0].shape[1]}")
        
        return J

    def getBinaryGrayCode(self, standard_size=None, size_method=None):
        '''
        将格雷码图像二值化处理
        
        该方法首先计算二值化阈值，然后对格雷码图像进行二值化处理。
        二值化后的图像便于后续的格雷码解码。
        
        参数:
            standard_size: 标准图像尺寸，如果为None则自动计算
            size_method: 尺寸调整方法，默认为"crop"
            
        返回:
            list: 二值化后的格雷码图像列表，所有图像具有相同的尺寸
        '''
        # 更新标准尺寸和方法
        if standard_size is not None:
            self.standard_size = standard_size
        if size_method is not None:
            self.size_method = size_method
            
        threshold = self.get_threshold(standard_size=self.standard_size, size_method=self.size_method)  # 计算二值化阈值
        graycodes = self.get_GC_images(self.standard_size, self.size_method)  # 获取格雷码图像
        
        rows, cols = threshold.shape
        # 逐像素进行二值化处理
        for a in range(len(graycodes)):
            for b in range(rows):
                for c in range(cols):
                    if graycodes[a][b,c] <= threshold[b,c]:
                        graycodes[a][b,c] = 0  # 低于阈值设为0
                    else:
                        graycodes[a][b,c] = 255  # 高于阈值设为255
        
        return graycodes


# 整合 UnwrappedPhase 类 (原 unwrapped_phase_algorithm.py)
class UnwrappedPhase():
    """
    解包裹相位计算类
    
    该类实现了基于格雷码辅助的相位解包裹算法。
    解包裹是将包裹相位（范围[0, 2π]）转换为连续绝对相位的过程。
    """
    
    def __init__(self, n: int = 5, direction: str = "horizontal"):
        """
        初始化解包裹相位计算器
        
        参数:
            n (int): 格雷码位数，默认为5
            direction (str): 条纹方向，可选值为"horizontal"、"vertical"或"both"，默认为"horizontal"
        """
        self.n = n
        self.direction = direction
        self.standard_size = None  # 标准图像尺寸
        self.size_method = "crop"  # 默认使用裁剪方法
    
    def getBinarizedGrayCodes(self, m: int = 5):
        """
        获得二值化后的格雷码图像
        
        该方法读取已经二值化处理的格雷码图像。如果找不到图像文件，
        会根据解包裹方向生成相应的测试用格雷码图像。二值化格雷码图像是解包裹相位
        过程中确定条纹周期的关键。
        
        参数:
            m (int): 格雷码位数，默认为5
            
        返回:
            list: 二值化后的格雷码图像列表，每个元素是一个二值化图像（0或1）
        """
        BGC = []
        # 根据解包裹方向确定要使用的格雷码类型
        is_vertical_unwrap = (self.direction == "vertical")
        
        # 设置基本文件名模式
        if is_vertical_unwrap:
            print("准备进行垂直方向解包裹，使用水平条纹格雷码...")
            base_filename = r"gray_patterns_horizontal\matched_binary_"
        else:
            print("准备进行水平方向解包裹，使用垂直条纹格雷码...")
            base_filename = r"gray_patterns\matched_binary_"
        
        for i in range(self.n):
            try:
                # 尝试读取已有的二值化格雷码图像
                filename = base_filename + str(i) + ".png"
                
                # 尝试直接读取图像
                img = cv.imread(filename, 0)
                if img is None:
                    # 如果直接读取失败，尝试使用二进制方式读取
                    img_file = np.fromfile(filename, dtype=np.uint8)
                    img = cv.imdecode(img_file, 0)
                
                if img is None:
                    raise FileNotFoundError(f"无法读取图像文件: {filename}")
            except Exception as e:
                print(f"读取二值化格雷码图像 {filename} 失败: {e}")
                print(f"生成{'水平' if is_vertical_unwrap else '垂直'}条纹格雷码测试图像...")
                
                # 创建一个格雷码生成器
                g = GrayCode(self.n)
                
                # 生成合适方向的格雷码图案
                if is_vertical_unwrap:
                    # 为垂直方向解包裹生成水平条纹格雷码
                    pattern = np.zeros((480, 640), np.uint8)
                    row = g.codes[i, :]
                    per_row = int(480 / len(row))
                    # 按行填充，创建水平条纹
                    for j in range(len(row)):
                        pattern[j * per_row: (j + 1) * per_row, :] = row[j] * 255
                else:
                    # 为水平方向解包裹生成垂直条纹格雷码
                    pattern = g.toPattern(i, 640, 480)
                
                # 模拟二值化后的图像
                _, img = cv.threshold(pattern, 127, 255, cv.THRESH_BINARY)
                
                # 保存生成的图像，以便下次使用
                save_dir = "gray_patterns_horizontal" if is_vertical_unwrap else "gray_patterns"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                save_filename = os.path.join(save_dir, f"matched_binary_{i}.png")
                cv.imwrite(save_filename, img)
                print(f"已生成并保存{('水平' if is_vertical_unwrap else '垂直')}条纹格雷码图像: {save_filename}")
                
                # 同时保存原始格雷码图像（非二值化）
                gray_filename = os.path.join(save_dir, f"gray_bit_{i}.png")
                cv.imwrite(gray_filename, pattern)
                print(f"已生成并保存原始{('水平' if is_vertical_unwrap else '垂直')}条纹格雷码图像: {gray_filename}")
            
            # 将像素值从[0,255]归一化到[0,1]范围
            img_scaled = img/255
            
            # 转换为uint8类型，此时像素值为0或1
            BGC.append(img_scaled.astype(np.uint8))
        
        # 标准化所有格雷码图像的尺寸
        BGC = normalize_image_size(BGC, self.standard_size, self.size_method)
        
        # 打印图像尺寸信息
        if len(BGC) > 0 and BGC[0] is not None:
            print(f"二值化格雷码图像尺寸: {BGC[0].shape[0]}x{BGC[0].shape[1]}")
            print(f"解包裹方向: {self.direction}")
            print(f"使用的格雷码类型: {'水平条纹' if is_vertical_unwrap else '垂直条纹'}")
            
            # 更新标准尺寸（如果之前未设置）
            if self.standard_size is None:
                self.standard_size = BGC[0].shape[:2]
                print(f"设置标准图像尺寸为: {self.standard_size[0]}x{self.standard_size[1]}")
        
        return BGC

    def computeUnwrappedPhase(self, show_details=True, direction=None, standard_size=None, size_method=None):
        """
        计算解包裹相位（改进版）
        
        该方法整合了整个解包裹相位流程，包括：
        1. 获取四步相移图像
        2. 计算包裹相位
        3. 获取k1和k2矩阵（格雷码解码）
        4. 使用相位连续性约束进行解包裹
        5. 评估相位质量
        6. 检测相位跳变
        7. 平滑处理
        8. 优化相位跳变区域
        9. 生成和保存结果
        
        该方法整合了多种改进算法，提高解包裹相位的质量：
        - 相位连续性约束
        - 相位质量评估
        - 相位平滑处理
        - 相位跳变区域优化
        
        参数:
            show_details: 是否显示详细的中间过程
            direction: 解包裹方向，可选值为"horizontal"、"vertical"。如果为None，则使用初始化时设置的方向
            standard_size: 标准图像尺寸，如果为None则自动计算最小尺寸
            size_method: 尺寸调整方法，可选值为"crop"或"resize"
        
        返回:
            numpy.ndarray: 最终的解包裹相位矩阵
        """
        # 更新标准尺寸和方法（如果指定）
        if standard_size is not None:
            self.standard_size = standard_size
        if size_method is not None:
            self.size_method = size_method
            
        # 如果没有指定方向，使用初始化时设置的方向
        if direction is None:
            direction = self.direction
        else:
            # 如果指定了新方向，更新实例的方向属性
            self.direction = direction
            
        print(f"\n开始计算{direction}方向上的解包裹相位...")
        
        # 检查必要的目录是否存在
        directories_exist = os.path.exists("fringe_patterns")
        gray_patterns_dir = "gray_patterns_horizontal" if direction == "vertical" else "gray_patterns"
        if not os.path.exists(gray_patterns_dir):
            directories_exist = False
            
        if not directories_exist:
            print(f"警告: 未找到所需的图像目录，将根据解包裹方向生成测试图案")
            self.generate_test_patterns()
            
        # 创建包裹相位计算器实例
        WP = WrappedPhase()
        
        # 获取四步相移图像，应用标准尺寸
        fringe_images = WP.getImageData(direction=direction, standard_size=self.standard_size, size_method=self.size_method)
        
        # 如果未设置标准尺寸，使用相移图像的尺寸作为标准
        if self.standard_size is None and len(fringe_images) > 0 and fringe_images[0] is not None:
            self.standard_size = fringe_images[0].shape[:2]
            print(f"设置标准图像尺寸为: {self.standard_size[0]}x{self.standard_size[1]}")
        
        # 计算包裹相位
        wrapped_pha = WP.computeWrappedphase(fringe_images)
        
        # 打印包裹相位的范围
        print(f"包裹相位范围: [{np.min(wrapped_pha)}, {np.max(wrapped_pha)}]")
        print(f"理论包裹相位范围: [0, {2*math.pi}]")
        
        # 获得k1和k2矩阵
        k1, k2 = self.get_k1_k2()
        
        # 检查包裹相位和k1、k2的尺寸是否一致
        print(f"\n包裹相位尺寸: {wrapped_pha.shape}")
        print(f"k1尺寸: {k1.shape}")
        print(f"k2尺寸: {k2.shape}")
        
        # 断言检查尺寸是否一致
        try:
            assert wrapped_pha.shape == k1.shape == k2.shape, "包裹相位和k1、k2的尺寸不一致!"
        except AssertionError as e:
            print(f"错误: {e}")
            print("尝试调整尺寸...")
            
            # 如果尺寸不一致，尝试调整到相同的尺寸
            min_rows = min(wrapped_pha.shape[0], k1.shape[0], k2.shape[0])
            min_cols = min(wrapped_pha.shape[1], k1.shape[1], k2.shape[1])
            
            wrapped_pha = wrapped_pha[:min_rows, :min_cols]
            k1 = k1[:min_rows, :min_cols]
            k2 = k2[:min_rows, :min_cols]
            
            print(f"调整后尺寸: {min_rows}x{min_cols}")
            
            # 更新标准尺寸
            self.standard_size = (min_rows, min_cols)
            print(f"更新标准图像尺寸为: {min_rows}x{min_cols}")
        
        # 1. 使用相位连续性约束的解包裹算法
        print("\n步骤1: 使用相位连续性约束的解包裹算法...")
        unwrapped_pha = self.phase_unwrapping_with_continuity(wrapped_pha, k1, k2)
        
        # 限制解包裹相位的范围
        max_phase = 32 * math.pi  # 5位格雷码的理论最大值
        unwrapped_pha = np.clip(unwrapped_pha, 0, max_phase)
        
        # 打印解包裹相位的范围
        print(f"连续性约束解包裹相位的范围: [{np.min(unwrapped_pha)}, {np.max(unwrapped_pha)}]")
        print(f"理论范围: [0, {32*math.pi}]")
        
        # 如果不需要显示详细过程，直接返回结果
        if not show_details:
            return unwrapped_pha
        
        # 2. 评估相位质量
        print("\n步骤2: 评估相位质量...")
        quality = self.estimate_phase_quality(fringe_images, show_details)
        
        # 3. 可视化相位跳变区域
        print("\n步骤3: 可视化相位跳变区域...")
        jumps = self.visualize_phase_jumps(unwrapped_pha, threshold=0.5, show_details=show_details)
        
        # 4. 对解包裹相位进行平滑处理
        print("\n步骤4: 对解包裹相位进行平滑处理...")
        smoothed_pha = self.smooth_unwrapped_phase(unwrapped_pha)
        
        # 5. 可视化平滑后的相位跳变区域
        print("\n步骤5: 可视化平滑后的相位跳变区域...")
        smoothed_jumps = self.visualize_phase_jumps(smoothed_pha, threshold=0.5, show_details=show_details)
        
        # 比较平滑前后的跳变点数量
        jumps_before = np.sum(jumps > 0)
        jumps_after = np.sum(smoothed_jumps > 0)
        if jumps_before > 0:
            reduction_ratio = (jumps_before - jumps_after) / jumps_before * 100
        else:
            reduction_ratio = 0
            
        print(f"\n平滑处理效果:")
        print(f"  平滑前跳变点数量: {jumps_before}")
        print(f"  平滑后跳变点数量: {jumps_after}")
        print(f"  跳变点减少比例: {reduction_ratio:.2f}%")
        
        # 6. 综合质量评估，生成最终解包裹相位
        print("\n步骤6: 综合质量评估，生成最终解包裹相位...")
        
        # 使用质量图作为权重，对低质量区域进行特殊处理
        low_quality_mask = quality < 0.2
        
        # 对低质量区域使用更大的平滑核
        final_pha = smoothed_pha.copy()
        if np.any(low_quality_mask):
            # 对低质量区域使用更大的平滑核
            large_kernel_size = 9
            extra_smoothed = cv.GaussianBlur(smoothed_pha, (large_kernel_size, large_kernel_size), 0)
            final_pha[low_quality_mask] = extra_smoothed[low_quality_mask]
            
            print(f"  对{np.sum(low_quality_mask)}个低质量像素点进行了额外平滑处理")
        
        # 7. 优化相位跳变区域
        print("\n步骤7: 优化相位跳变区域...")
        final_pha = self.optimize_phase_jumps(final_pha, smoothed_jumps, quality, show_details)
        
        # 打印最终解包裹相位的范围
        print(f"最终解包裹相位的范围: [{np.min(final_pha)}, {np.max(final_pha)}]")
        
        # 保存最终结果
        final_scaled = (final_pha * 255 / np.max(final_pha)).astype(np.uint8)
        final_color = cv.applyColorMap(final_scaled, cv.COLORMAP_JET)
        
        # 创建results目录（如果不存在）
        if not os.path.exists("results"):
            os.makedirs("results")
        
        # 根据方向保存结果
        if direction == "horizontal":
            result_filename = "results/horizontal_unwrapped_phase.png"
        else:
            result_filename = "results/vertical_unwrapped_phase.png"
            
        cv.imwrite(result_filename, final_color)
        print(f"解包裹相位结果已保存到: {result_filename}")
        
        return final_pha

    def estimate_phase_quality(self, fringe_images, show_details=True):
        """
        估计相位质量
        
        该方法通过计算调制度来评估每个像素点的相位质量。
        调制度是衡量相位质量的重要指标，它反映了条纹对比度和信噪比。
        调制度越高，相位质量越好。
        
        调制度计算公式: sqrt((I3-I1)^2 + (I0-I2)^2) / (I0+I1+I2+I3)
        
        参数:
            fringe_images: 四步相移的条纹图像列表 [I0, I1, I2, I3]
            show_details: 是否显示质量评估的详细信息和可视化结果
            
        返回:
            numpy.ndarray: 相位质量矩阵，值范围[0,1]，值越大表示质量越好
        """
        # 确保输入图像是float32类型
        i0, i1, i2, i3 = [img.astype(np.float32) for img in fringe_images]
        
        # 计算调制度
        # 调制度公式: sqrt((I3-I1)^2 + (I0-I2)^2) / (I0+I1+I2+I3)
        numerator = np.sqrt((i3 - i1)**2 + (i0 - i2)**2)
        denominator = i0 + i1 + i2 + i3
        modulation = numerator / (denominator + 1e-6)  # 添加小常数避免除零
        
        # 归一化到[0,1]范围
        modulation = np.clip(modulation, 0, 1)
        
        # 保存质量图像
        if not os.path.exists("results"):
            os.makedirs("results")
        
        # 将调制度缩放到[0,255]范围
        quality_scaled = (modulation * 255).astype(np.uint8)
        
        # 应用伪彩色映射以增强可视化效果
        quality_color = cv.applyColorMap(quality_scaled, cv.COLORMAP_JET)
        
        # 保存质量图像
        cv.imwrite("results/phase_quality.png", quality_color)
        
        # 计算质量统计信息
        quality_mean = np.mean(modulation)
        quality_std = np.std(modulation)
        quality_min = np.min(modulation)
        quality_max = np.max(modulation)
        
        print(f"\n相位质量统计:")
        print(f"  平均质量: {quality_mean:.4f}")
        print(f"  质量标准差: {quality_std:.4f}")
        print(f"  最小质量: {quality_min:.4f}")
        print(f"  最大质量: {quality_max:.4f}")
        
        # 计算低质量区域的比例
        low_quality_threshold = 0.2
        low_quality_ratio = np.sum(modulation < low_quality_threshold) / modulation.size
        print(f"  低质量区域比例 (< {low_quality_threshold}): {low_quality_ratio*100:.2f}%")
        
        # 如果需要显示详细信息，则显示质量图像
        if show_details:
            cv.imshow("Phase Quality", quality_color)
            print("正在显示相位质量图，按任意键继续...")
            cv.waitKey(0)
            cv.destroyWindow("Phase Quality")
        
        return modulation

    def visualize_phase_jumps(self, unwrapped_pha, threshold=0.5, show_details=True):
        """
        可视化相位跳变区域
        
        该方法检测相位梯度过大的区域，识别可能的相位跳变点。
        相位跳变是解包裹相位中的常见问题，通常表现为相邻像素之间
        的相位值差异异常大。
        
        算法流程：
        1. 计算水平和垂直方向的相位梯度
        2. 计算梯度幅值
        3. 根据阈值检测相位跳变区域
        4. 统计和可视化跳变区域
        
        参数:
            unwrapped_pha: 解包裹相位矩阵
            threshold: 相位跳变阈值，默认为0.5（相位梯度大于0.5*pi被认为是跳变）
            show_details: 是否显示相位跳变的可视化结果
            
        返回:
            numpy.ndarray: 相位跳变区域的二值图像，跳变区域为255，非跳变区域为0
        """
        # 计算水平和垂直方向的相位梯度
        grad_x = np.abs(np.diff(unwrapped_pha, axis=1, append=unwrapped_pha[:, :1]))
        grad_y = np.abs(np.diff(unwrapped_pha, axis=0, append=unwrapped_pha[:1, :]))
        
        # 计算梯度幅值
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 检测相位跳变区域
        jump_threshold = threshold * math.pi
        jumps = (grad_magnitude > jump_threshold).astype(np.uint8) * 255
        
        # 计算跳变点的数量和比例
        jump_count = np.sum(jumps > 0)
        jump_ratio = jump_count / unwrapped_pha.size
        
        print(f"\n相位跳变分析:")
        print(f"  跳变阈值: {jump_threshold:.4f} rad")
        print(f"  跳变点数量: {jump_count}")
        print(f"  跳变点比例: {jump_ratio*100:.4f}%")
        
        # 保存二值化跳变图像
        if not os.path.exists("results"):
            os.makedirs("results")
        cv.imwrite("results/phase_jumps_binary.png", jumps)
        
        # 创建彩色跳变图像，红色表示跳变区域
        jumps_color = np.zeros((jumps.shape[0], jumps.shape[1], 3), dtype=np.uint8)
        jumps_color[jumps > 0] = [0, 0, 255]  # 红色表示跳变区域
        cv.imwrite("results/phase_jumps_color.png", jumps_color)
        
        # 在原始相位图上叠加跳变区域
        # 将相位缩放到[0,255]范围
        phase_scaled = (unwrapped_pha * 255 / np.max(unwrapped_pha)).astype(np.uint8)
        phase_color = cv.applyColorMap(phase_scaled, cv.COLORMAP_JET)
        
        # 叠加跳变区域
        overlay = phase_color.copy()
        overlay[jumps > 0] = [0, 0, 255]  # 红色表示跳变区域
        
        # 使用alpha混合
        alpha = 0.7
        phase_with_jumps = cv.addWeighted(phase_color, alpha, overlay, 1-alpha, 0)
        cv.imwrite("results/phase_with_jumps.png", phase_with_jumps)
        
        # 如果需要显示详细信息，则显示跳变可视化结果
        if show_details:
            cv.imshow("Phase Jumps", jumps_color)
            cv.imshow("Phase with Jumps", phase_with_jumps)
            print("正在显示相位跳变图，按任意键继续...")
            cv.waitKey(0)
            cv.destroyWindow("Phase Jumps")
            cv.destroyWindow("Phase with Jumps")
        
        return jumps

    def optimize_phase_jumps(self, unwrapped_pha, jumps, quality, show_details=True):
        """
        进一步优化相位跳变区域
        
        该方法针对检测到的相位跳变区域进行特殊处理，
        通过考虑周围高质量区域的相位值，修复跳变区域的相位值，
        提高相位的连续性和准确性。
        
        算法流程：
        1. 识别跳变区域的像素点
        2. 对每个跳变点，考虑其周围区域的相位值和质量值
        3. 使用质量加权平均计算新的相位值
        4. 使用中值滤波进一步平滑跳变区域
        5. 评估优化效果
        
        参数:
            unwrapped_pha: 解包裹相位矩阵
            jumps: 相位跳变区域的二值图像
            quality: 相位质量矩阵
            show_details: 是否显示优化过程的可视化结果
            
        返回:
            numpy.ndarray: 优化后的相位矩阵
        """
        # 创建优化后的相位矩阵
        optimized_pha = unwrapped_pha.copy()
        
        # 获取跳变区域的坐标
        jump_coords = np.where(jumps > 0)
        
        if len(jump_coords[0]) == 0:
            print("没有检测到相位跳变区域，无需优化")
            return optimized_pha
            
        print(f"检测到{len(jump_coords[0])}个相位跳变点，正在优化...")
        
        # 对每个跳变点进行处理
        for i in range(len(jump_coords[0])):
            y, x = jump_coords[0][i], jump_coords[1][i]
            
            # 获取周围区域（7x7窗口）
            y_min = max(0, y - 3)
            y_max = min(unwrapped_pha.shape[0] - 1, y + 3)
            x_min = max(0, x - 3)
            x_max = min(unwrapped_pha.shape[1] - 1, x + 3)
            
            # 提取周围区域的相位值和质量值
            region_phase = unwrapped_pha[y_min:y_max+1, x_min:x_max+1]
            region_quality = quality[y_min:y_max+1, x_min:x_max+1]
            region_jumps = jumps[y_min:y_max+1, x_min:x_max+1]
            
            # 排除跳变点本身
            mask = region_jumps == 0
            if not np.any(mask):
                continue  # 如果周围全是跳变点，则跳过
                
            # 使用质量作为权重，计算加权平均
            weights = region_quality * mask
            if np.sum(weights) > 0:
                weighted_phase = np.sum(region_phase * weights) / np.sum(weights)
                optimized_pha[y, x] = weighted_phase
        
        # 使用中值滤波进一步平滑跳变区域
        for i in range(len(jump_coords[0])):
            y, x = jump_coords[0][i], jump_coords[1][i]
            
            # 获取周围区域（5x5窗口）
            y_min = max(0, y - 2)
            y_max = min(optimized_pha.shape[0] - 1, y + 2)
            x_min = max(0, x - 2)
            x_max = min(optimized_pha.shape[1] - 1, x + 2)
            
            # 提取周围区域的相位值
            region_phase = optimized_pha[y_min:y_max+1, x_min:x_max+1].flatten()
            
            # 计算中值
            median_phase = np.median(region_phase)
            
            # 如果当前值与中值相差太大，则替换为中值
            if abs(optimized_pha[y, x] - median_phase) > math.pi:
                optimized_pha[y, x] = median_phase
        
        # 计算优化前后的差异
        diff = np.abs(unwrapped_pha - optimized_pha)
        
        # 将差异缩放到[0,255]范围用于显示
        diff_scaled = (diff * 255 / np.max(diff)).astype(np.uint8) if np.max(diff) > 0 else np.zeros_like(diff, dtype=np.uint8)
        
        # 应用伪彩色映射以增强可视化效果
        diff_color = cv.applyColorMap(diff_scaled, cv.COLORMAP_JET)
        
        # 保存差异图像
        cv.imwrite("results/optimization_difference.png", diff_color)
        
        # 可视化优化后的相位
        optimized_scaled = (optimized_pha * 255 / np.max(optimized_pha)).astype(np.uint8)
        optimized_color = cv.applyColorMap(optimized_scaled, cv.COLORMAP_JET)
        cv.imwrite("results/optimized_phase.png", optimized_color)
        
        # 检查优化后的相位跳变
        optimized_jumps = self.visualize_phase_jumps(optimized_pha, threshold=0.5, show_details=False)
        
        # 比较优化前后的跳变点数量
        jumps_before = np.sum(jumps > 0)
        jumps_after = np.sum(optimized_jumps > 0)
        if jumps_before > 0:
            reduction_ratio = (jumps_before - jumps_after) / jumps_before * 100
        else:
            reduction_ratio = 0
        
        print(f"\n优化处理效果:")
        print(f"  优化前跳变点数量: {jumps_before}")
        print(f"  优化后跳变点数量: {jumps_after}")
        print(f"  跳变点减少比例: {reduction_ratio:.2f}%")
        print(f"  最大相位差异: {np.max(diff):.6f} rad")
        print(f"  平均相位差异: {np.mean(diff):.6f} rad")
        
        # 如果需要显示详细信息，则显示优化效果
        if show_details:
            cv.imshow("Optimization Difference", diff_color)
            cv.imshow("Optimized Phase", optimized_color)
            print("正在显示优化效果，按任意键继续...")
            cv.waitKey(0)
            cv.destroyWindow("Optimization Difference")
            cv.destroyWindow("Optimized Phase")
        
        return optimized_pha

    def show_graycodes(self):
        """显示格雷码图像，用于调试和可视化"""
        try:
            # 读取和显示原始格雷码图像
            print("显示原始格雷码图像...")
            for i in range(self.n):
                filename = f"gray_patterns/gray_bit_{i}.png"
                img = cv.imread(filename, 0)
                if img is not None:
                    cv.imshow(f"Gray Code {i}", img)
                    cv.waitKey(800)  # 显示800毫秒
                else:
                    print(f"无法读取图像 {filename}")
            
            # 读取和显示二值化格雷码图像
            print("显示二值化格雷码图像...")
            for i in range(self.n):
                filename = f"gray_patterns/matched_binary_{i}.png"
                img = cv.imread(filename, 0)
                if img is not None:
                    cv.imshow(f"Binary Gray Code {i}", img)
                    cv.waitKey(800)  # 显示800毫秒
                else:
                    print(f"无法读取图像 {filename}")
            
            # 等待用户按键关闭所有窗口
            print("按任意键继续...")
            cv.waitKey(0)
            cv.destroyAllWindows()
            
        except Exception as e:
            print(f"显示格雷码图像时出错: {e}")
            cv.destroyAllWindows()
    
    def show_fringe_patterns(self):
        """显示四步相移条纹图像，用于调试和可视化"""
        try:
            wp = WrappedPhase()
            fringe_images = wp.getImageData()
            
            print("显示四步相移条纹图像...")
            for i, img in enumerate(fringe_images):
                cv.imshow(f"Fringe Pattern {i} (Phase shift: {i*90}°)", img)
                cv.waitKey(800)  # 显示800毫秒
            
            # 等待用户按键关闭所有窗口
            print("按任意键继续...")
            cv.waitKey(0)
            cv.destroyAllWindows()
            
        except Exception as e:
            print(f"显示四步相移图像时出错: {e}")
            cv.destroyAllWindows()
    
    def showUnwrappedPhase(self, show_k1_k2=True, show_details=True, direction=None):
        """
        显示解包裹相位
        
        该方法计算并显示解包裹相位结果，同时提供交互式界面，
        允许用户查看相位值、保存结果等。
        
        该方法的主要功能包括：
        1. 调用computeUnwrappedPhase计算解包裹相位
        2. 可选地显示k1和k2解码结果
        3. 生成多种可视化效果（原始、伪彩色、直方图均衡化等）
        4. 提供鼠标交互功能，显示点击位置的相位值
        5. 保存结果到指定目录
        
        参数:
            show_k1_k2: 是否显示k1和k2解码图像
            show_details: 是否显示详细的中间过程图像
            direction: 解包裹方向，可选值为"horizontal"、"vertical"。如果为None，则使用初始化时设置的方向
            
        返回:
            numpy.ndarray: 解包裹相位矩阵
        """
        # 如果没有指定方向，使用初始化时设置的方向
        if direction is None:
            direction = self.direction
            
        # 计算解包裹相位（使用改进的算法）
        upha = self.computeUnwrappedPhase(show_details, direction)
        
        # 根据方向确定显示标题
        if direction == "horizontal":
            direction_text = "水平方向"
        elif direction == "vertical":
            direction_text = "垂直方向"
        elif direction == "both":
            direction_text = "组合"
        else:
            direction_text = ""
        
        # 如果需要显示k1和k2解码结果
        if show_k1_k2:
            # 获取k1和k2矩阵
            k1, k2 = self.get_k1_k2()
            
            # 显示k1图像（4位格雷码解码结果）
            # 将k1缩放到[0,255]以便显示
            k1_vis = (k1 * (255/15)).astype(np.uint8)  # 4位格雷码范围[0,15]
            cv.imshow("k1 matrix (4-bit Gray code)", k1_vis)
            
            # 显示k2图像（5位格雷码解码结果）
            # 将k2缩放到[0,255]以便显示
            k2_vis = (k2 * (255/31)).astype(np.uint8)  # 5位格雷码范围[0,31]
            cv.imshow("k2 matrix (5-bit Gray code)", k2_vis)
            
            # 等待用户按键
            print("正在显示k1和k2解码结果，按任意键继续...")
            cv.waitKey(0)
            cv.destroyAllWindows()
        
        # 将相位值缩放到[0,255]范围用于显示
        # 对于5位格雷码，相位范围约为[0, 32π]
        upha_scaled = np.rint(upha*255/(32*math.pi))
        
        # 转换为uint8类型用于图像显示
        upha_scaled_uint = upha_scaled.astype(np.uint8)
        
        # 创建results目录（如果不存在）
        if not os.path.exists("results"):
            os.makedirs("results")
        
        # 保存不同视图的解包裹相位图像
        # 1. 原始缩放视图
        cv.imwrite(f"results/{direction}_unwrapped_phase_original.png", upha_scaled_uint)
        
        # 2. 应用伪彩色映射以增强可视化效果
        upha_color = cv.applyColorMap(upha_scaled_uint, cv.COLORMAP_JET)
        cv.imwrite(f"results/{direction}_unwrapped_phase_color.png", upha_color)
        
        # 3. 应用直方图均衡化以增强对比度
        upha_eq = cv.equalizeHist(upha_scaled_uint)
        cv.imwrite(f"results/{direction}_unwrapped_phase_equalized.png", upha_eq)
        
        # 4. 3D可视化（保存为高度图）
        # 将相位值归一化到[0,1]范围
        upha_norm = upha / np.max(upha)
        # 保存为16位PNG，以保留更多细节
        cv.imwrite(f"results/{direction}_unwrapped_phase_height.png", (upha_norm * 65535).astype(np.uint16))
        
        # 定义鼠标回调函数，显示点击位置的相位值
        def mouse_callback(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                # 获取点击位置的相位值
                phase_value = upha[y, x]
                # 计算相位值对应的周期数
                period = phase_value / (2 * math.pi)
                # 在控制台显示信息
                print(f"点击位置 ({x}, {y}):")
                print(f"  相位值: {phase_value:.6f} rad")
                print(f"  周期数: {period:.6f}")
                print(f"  灰度值: {upha_scaled_uint[y, x]}")
                
                # 在图像上显示相位值
                # 创建一个副本以避免修改原图
                display_img = upha_color.copy()
                # 绘制十字标记
                cv.drawMarker(display_img, (x, y), (0, 255, 255), cv.MARKER_CROSS, 20, 2)
                # 添加文本
                text = f"Phase: {phase_value:.2f} rad"
                cv.putText(display_img, text, (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                # 更新显示
                cv.imshow(f"{direction_text}解包裹相位 (彩色)", display_img)
        
        # 显示解包裹相位图像
        cv.imshow(f"{direction_text}解包裹相位", upha_scaled_uint)
        cv.imshow(f"{direction_text}解包裹相位 (彩色)", upha_color)
        cv.imshow(f"{direction_text}解包裹相位 (均衡化)", upha_eq)
        
        # 设置鼠标回调
        cv.setMouseCallback(f"{direction_text}解包裹相位 (彩色)", mouse_callback)
        
        # 显示交互提示
        print("\n交互提示:")
        print("- 鼠标点击彩色图像可显示该点的相位值")
        print("- 按's'键保存图像")
        print("- 按'q'键或ESC键退出")
        
        # 等待用户按键
        while True:
            key = cv.waitKey(0)
            if key == ord("s"):
                # 按's'键保存图像
                cv.imwrite(f"results/{direction}_Absolute_pha.png", upha_scaled_uint)
                print(f"已保存图像: results/{direction}_Absolute_pha.png")
            elif key == ord("q") or key == 27:  # 'q'键或ESC键
                break
        
        # 关闭所有图像窗口
        cv.destroyAllWindows()
        
        print(f"{direction_text}解包裹相位图像已保存到results目录")
        return upha

    # 添加其他需要的方法...
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
        以减少噪声和相位跳变。中值滤波主要用于去除离群值（如相位跳变点），
        而高斯滤波则用于保留相位的整体结构，同时减少小的波动。
        
        参数:
            unwrapped_pha: 解包裹相位矩阵
            kernel_size: 滤波核大小，默认为5
            
        返回:
            numpy.ndarray: 平滑后的相位矩阵
        """
        # 将相位矩阵转换为float32类型，确保滤波操作的精度
        phase_float32 = unwrapped_pha.astype(np.float32)
        
        # 使用中值滤波去除离群值（相位跳变点）
        # 中值滤波对于去除椒盐噪声（如相位跳变）非常有效
        median_filtered = cv.medianBlur(phase_float32, kernel_size)
        
        # 使用高斯滤波进一步平滑
        # 高斯滤波可以保留相位的整体结构，同时减少小的波动
        smoothed = cv.GaussianBlur(median_filtered, (kernel_size, kernel_size), 0)
        
        return smoothed
    
    def generate_test_patterns(self):
        """
        生成测试图案
        
        当找不到原始图像文件时，该方法会生成用于测试的四步相移图像和格雷码图像。
        这些测试图像模拟了真实的结构光扫描场景，包括：
        - 四步相移条纹图像（相移量分别为0, π/2, π, 3π/2）
        - 5位格雷码图像（水平和垂直两个方向）
        - 二值化后的格雷码图像
        
        生成的测试图像保存在相应的目录中，用于后续的相位解包裹处理。
        
        返回:
            bool: 生成成功返回True
        """
        print("开始生成测试图像...")
        
        # 创建目录
        for directory in ["fringe_patterns", "gray_patterns", "gray_patterns_horizontal", "results"]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"创建了{directory}目录")
            
        # 图像参数设置
        # 如果已设置标准尺寸，使用标准尺寸
        if self.standard_size is not None:
            height, width = self.standard_size
            print(f"使用标准尺寸 {width}x{height} 生成测试图像")
        else:
            width, height = 640, 480  # 默认测试图像大小
            print(f"使用默认尺寸 {width}x{height} 生成测试图像")
            
        freq = 15  # 空间频率，决定条纹的密度
        noise_level = 5  # 噪声级别，模拟真实成像的噪声（降低噪声以获得更清晰的条纹）
        
        # 生成相移条纹图案（四步相移法）
        print("生成四步相移条纹图案...")
        x = np.arange(0, width)
        y = np.arange(0, height).reshape(height, 1)
        
        # 创建网格，用于生成条纹
        [X, Y] = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
        
        # 生成标准的垂直条纹投影图案 (I1-I4)（用于水平方向解包裹）
        print("生成垂直条纹四步相移图像 (I1-I4)...")
        for i in range(4):
            phase_shift = i * math.pi/2  # 相移量：0, π/2, π, 3π/2
            
            # 生成纯正弦条纹（垂直条纹，沿X方向变化）- 没有Z表面调制
            fringe = 128 + 100 * np.sin(2 * math.pi * freq * X + phase_shift)
            
            # 添加极少的高斯噪声（仅用于模拟轻微投影误差）
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, fringe.shape)
                fringe = fringe + noise
            
            # 限制像素值范围到[0, 255]
            fringe = np.clip(fringe, 0, 255).astype(np.uint8)
            
            # 保存图像
            filename = f"fringe_patterns/I{i+1}.png"
            cv.imwrite(filename, fringe)
            print(f"生成了垂直条纹投影图像: {filename}")
        
        # 生成标准的水平条纹投影图案 (I5-I8)（用于垂直方向解包裹）
        print("生成水平条纹四步相移图像 (I5-I8)...")
        for i in range(4):
            phase_shift = i * math.pi/2  # 相移量：0, π/2, π, 3π/2
            
            # 生成纯正弦条纹（水平条纹，沿Y方向变化）- 没有Z表面调制
            fringe = 128 + 100 * np.sin(2 * math.pi * freq * Y + phase_shift)
            
            # 添加极少的高斯噪声（仅用于模拟轻微投影误差）
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, fringe.shape)
                fringe = fringe + noise
            
            # 限制像素值范围到[0, 255]
            fringe = np.clip(fringe, 0, 255).astype(np.uint8)
            
            # 保存图像
            filename = f"fringe_patterns/I{i+5}.png"
            cv.imwrite(filename, fringe)
            print(f"生成了水平条纹投影图像: {filename}")
        
        # 为了演示目的，额外生成物体表面反射的模拟图像
        print("额外生成模拟的物体表面反射图像...")
        # 创建一个非平面的对象表面（模拟复杂形状）
        Z = 0.5 * (X**2 + Y**2) + 0.1 * np.sin(5*X) * np.cos(5*Y)  # 二次曲面加上正弦调制
        Z = Z / np.max(Z) * 2 * math.pi  # 归一化到[0, 2π]范围
        
        # 保存到单独目录
        if not os.path.exists("simulated_captured"):
            os.makedirs("simulated_captured")
            
        for i in range(4):
            phase_shift = i * math.pi/2
            # 垂直条纹在物体表面的反射（水平解包裹）
            reflected = 128 + 70 * np.sin(2 * math.pi * freq * X + Z + phase_shift)
            noise = np.random.normal(0, 10, reflected.shape)
            reflected = np.clip(reflected + noise, 0, 255).astype(np.uint8)
            cv.imwrite(f"simulated_captured/reflected_vertical_{i+1}.png", reflected)
            
            # 水平条纹在物体表面的反射（垂直解包裹）
            reflected = 128 + 70 * np.sin(2 * math.pi * freq * Y + Z + phase_shift)
            noise = np.random.normal(0, 10, reflected.shape)
            reflected = np.clip(reflected + noise, 0, 255).astype(np.uint8)
            cv.imwrite(f"simulated_captured/reflected_horizontal_{i+1}.png", reflected)
        
        # 确定当前的解包裹方向
        current_direction = self.direction
        print(f"当前解包裹方向: {current_direction}")
        
        # 生成垂直条纹格雷码图案（用于水平方向解包裹）
        print("生成垂直条纹格雷码图案（用于水平方向解包裹）...")
        g = GrayCode(5)  # 创建5位格雷码生成器
        
        for i in range(5):
            # 生成垂直条纹格雷码图案
            pattern = g.toPattern(i, width, height)
            
            # 为格雷码图案添加噪声和模糊（模拟真实成像）
            noise = np.random.normal(0, 5, pattern.shape)
            pattern_noisy = np.clip(pattern + noise, 0, 255).astype(np.uint8)
            
            # 添加轻微高斯模糊（模拟光学系统）
            pattern_blurred = cv.GaussianBlur(pattern_noisy, (3, 3), 0.5)
            
            # 保存原始格雷码图案
            cv.imwrite(f"gray_patterns/gray_bit_{i}.png", pattern_blurred)
            
            # 保存二值化格雷码图案（模拟二值化处理后的结果）
            _, binary_pattern = cv.threshold(pattern_blurred, 127, 255, cv.THRESH_BINARY)
            cv.imwrite(f"gray_patterns/matched_binary_{i}.png", binary_pattern)
            
            print(f"生成了垂直条纹格雷码图案 {i+1}/5")
        
        # 生成水平条纹格雷码图案（用于垂直方向解包裹）
        print("生成水平条纹格雷码图案（用于垂直方向解包裹）...")
        
        for i in range(5):
            # 生成水平条纹格雷码图案
            # 注意：为了生成水平条纹，我们需要转置格雷码图案的结构
            pattern_vertical = np.zeros((height, width), np.uint8)
            row = g.codes[i, :]
            per_row = int(height / len(row))
            
            # 按行填充，创建水平条纹
            for j in range(len(row)):
                pattern_vertical[j * per_row: (j + 1) * per_row, :] = row[j] * 255
            
            # 为格雷码图案添加噪声和模糊（模拟真实成像）
            noise = np.random.normal(0, 5, pattern_vertical.shape)
            pattern_noisy = np.clip(pattern_vertical + noise, 0, 255).astype(np.uint8)
            
            # 添加轻微高斯模糊（模拟光学系统）
            pattern_blurred = cv.GaussianBlur(pattern_noisy, (3, 3), 0.5)
            
            # 保存原始水平条纹格雷码图案
            cv.imwrite(f"gray_patterns_horizontal/gray_bit_{i}.png", pattern_blurred)
            
            # 保存二值化水平条纹格雷码图案
            _, binary_pattern = cv.threshold(pattern_blurred, 127, 255, cv.THRESH_BINARY)
            cv.imwrite(f"gray_patterns_horizontal/matched_binary_{i}.png", binary_pattern)
            
            print(f"生成了水平条纹格雷码图案 {i+1}/5")
            
            # 如果当前是垂直方向解包裹，还需要将水平条纹图像复制到gray_patterns目录下
            if current_direction == "vertical":
                cv.imwrite(f"gray_patterns/gray_bit_{i}.png", pattern_blurred)
                cv.imwrite(f"gray_patterns/matched_binary_{i}.png", binary_pattern)
                print(f"复制水平条纹图像到gray_patterns目录，用于垂直方向解包裹")
        
        # 保存尺寸信息
        if self.standard_size is None:
            self.standard_size = (height, width)
            
        print("测试图案生成完成!")
        print(f"所有测试图像尺寸: {width}x{height}")
        print("垂直条纹格雷码（用于水平解包裹）保存在：gray_patterns/")
        print("水平条纹格雷码（用于垂直解包裹）保存在：gray_patterns_horizontal/")
        
        # 提示当前解包裹方向使用的格雷码类型
        if current_direction == "vertical":
            print("当前设置为垂直方向解包裹，使用的是水平条纹格雷码")
            print("已自动将水平条纹格雷码复制到gray_patterns目录")
        else:
            print("当前设置为水平方向解包裹，使用的是垂直条纹格雷码")
        
        print("\n注意：生成的fringe_patterns中的图像是用于投影的标准相移图案")
        print("      真实应用中，您需要采集这些图案投影到物体表面后的图像")
        print("      simulated_captured目录中包含了模拟的物体反射图像，仅供参考")
        
        return True

    def combine_horizontal_vertical_phases(self, horizontal_phase, vertical_phase):
        """
        组合水平和垂直方向的解包裹相位
        
        该方法将水平方向和垂直方向的解包裹相位组合成二维相位分布图，
        保留两个方向的完整信息，可用于3D重建。
        
        组合后的图像可以显示出每个点在X和Y方向上的相位值，形成一个
        二维的相位分布，通过颜色的变化直观地展示相位的变化情况。
        
        参数:
            horizontal_phase: 水平方向的解包裹相位矩阵
            vertical_phase: 垂直方向的解包裹相位矩阵
            
        返回:
            tuple: (combined_visual, combined_data)
                   combined_visual: 用于可视化的彩色图
                   combined_data: 包含两个方向相位值的双通道数据
        """
        # 确保两个相位矩阵的尺寸一致
        if horizontal_phase.shape != vertical_phase.shape:
            print("警告: 水平和垂直相位矩阵尺寸不一致，将调整到相同尺寸")
            min_rows = min(horizontal_phase.shape[0], vertical_phase.shape[0])
            min_cols = min(horizontal_phase.shape[1], vertical_phase.shape[1])
            
            horizontal_phase = horizontal_phase[:min_rows, :min_cols]
            vertical_phase = vertical_phase[:min_rows, :min_cols]
        
        # 创建双通道数据结构，同时保存水平和垂直方向的相位值
        combined_data = np.zeros((horizontal_phase.shape[0], horizontal_phase.shape[1], 2), dtype=np.float32)
        combined_data[:,:,0] = horizontal_phase  # 第一个通道存储水平方向的相位
        combined_data[:,:,1] = vertical_phase    # 第二个通道存储垂直方向的相位
        
        # 创建可视化结果目录
        if not os.path.exists("results"):
            os.makedirs("results")
            
        # 使用numpy的格式保存原始相位数据（不使用EXR格式）
        np.save("results/combined_phase_data.npy", combined_data)
        np.save("results/horizontal_phase.npy", horizontal_phase)
        np.save("results/vertical_phase.npy", vertical_phase)
        
        # 方法1：创建水平和垂直方向相位的伪彩色图
        # 归一化相位值
        h_max = np.max(horizontal_phase)
        v_max = np.max(vertical_phase)
        
        h_norm = horizontal_phase / h_max if h_max > 0 else horizontal_phase
        v_norm = vertical_phase / v_max if v_max > 0 else vertical_phase
        
        h_color = cv.applyColorMap((h_norm * 255).astype(np.uint8), cv.COLORMAP_JET)
        v_color = cv.applyColorMap((v_norm * 255).astype(np.uint8), cv.COLORMAP_JET)
        
        # 保存单独的相位图
        cv.imwrite("results/horizontal_phase_color.png", h_color)
        cv.imwrite("results/vertical_phase_color.png", v_color)
        
        # 方法2：创建二维相位分布图
        # 将水平和垂直相位归一化到[0,1]范围，然后使用HSV颜色空间表示
        # 色调(H)用于表示相位角度(atan2(v_norm, h_norm))
        # 饱和度(S)保持最大
        # 亮度(V)表示相位幅值(sqrt(h_norm^2 + v_norm^2))
        
        # 计算相位角度和幅值
        phase_angle = np.arctan2(v_norm, h_norm) # 范围[-pi, pi]
        phase_angle = (phase_angle + np.pi) / (2 * np.pi) # 归一化到[0,1]
        
        phase_magnitude = np.sqrt(h_norm**2 + v_norm**2)
        # 归一化幅值到[0,1]
        phase_magnitude = np.clip(phase_magnitude, 0, 1)
        
        # 创建HSV图像
        hsv_image = np.zeros((horizontal_phase.shape[0], horizontal_phase.shape[1], 3), dtype=np.float32)
        hsv_image[:,:,0] = phase_angle * 179 # H通道，范围[0,179]
        hsv_image[:,:,1] = 255              # S通道，最大饱和度
        hsv_image[:,:,2] = phase_magnitude * 255 # V通道，范围[0,255]
        
        # 转换HSV到BGR
        hsv_image = hsv_image.astype(np.uint8)
        phase_dist_map = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)
        
        # 保存二维相位分布图
        cv.imwrite("results/phase_distribution_map.png", phase_dist_map)
        
        # 方法3：创建网格化的相位图，显示相位等值线
        grid_size = 40  # 网格大小
        grid_image = np.zeros((horizontal_phase.shape[0], horizontal_phase.shape[1], 3), dtype=np.uint8)
        
        # 创建相位等值线
        for i in range(0, horizontal_phase.shape[0], grid_size):
            for j in range(0, horizontal_phase.shape[1], grid_size):
                # 在网格点上绘制小正方形
                size = 3
                x1, y1 = max(0, j-size), max(0, i-size)
                x2, y2 = min(horizontal_phase.shape[1]-1, j+size), min(horizontal_phase.shape[0]-1, i+size)
                
                # 红色表示水平相位，蓝色表示垂直相位
                h_phase_val = h_norm[i, j]
                v_phase_val = v_norm[i, j]
                
                # 在网格点上绘制颜色方块
                grid_image[y1:y2, x1:x2, 0] = 0  # B
                grid_image[y1:y2, x1:x2, 1] = 0  # G
                grid_image[y1:y2, x1:x2, 2] = int(h_phase_val * 255)  # R
        
        # 在图像上叠加等相位线
        h_contour_levels = np.linspace(0, 1, 10)
        v_contour_levels = np.linspace(0, 1, 10)
        
        # 复制一份彩色相位图用于叠加等相位线
        contour_image = phase_dist_map.copy()
        
        # 对水平相位绘制等值线（红色）
        for level in h_contour_levels:
            h_binary = (np.abs(h_norm - level) < 0.02).astype(np.uint8) * 255
            contours, _ = cv.findContours(h_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(contour_image, contours, -1, (0, 0, 255), 1)
        
        # 对垂直相位绘制等值线（蓝色）
        for level in v_contour_levels:
            v_binary = (np.abs(v_norm - level) < 0.02).astype(np.uint8) * 255
            contours, _ = cv.findContours(v_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(contour_image, contours, -1, (255, 0, 0), 1)
        
        # 保存等相位线图
        cv.imwrite("results/phase_contour_map.png", contour_image)
        
        # 方法4：创建3D可视化图像（顶视图）
        # 将水平相位映射到X轴，垂直相位映射到Y轴，形成一个2D平面
        plot_image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # 归一化相位到[0,1]
        h_phase_norm = h_norm
        v_phase_norm = v_norm
        
        # 对每个像素，根据其水平和垂直相位值，在plot_image上对应位置标记一个点
        for i in range(0, horizontal_phase.shape[0], 2):
            for j in range(0, horizontal_phase.shape[1], 2):
                x = int(h_phase_norm[i, j] * 511)
                y = int(v_phase_norm[i, j] * 511)
                x = min(max(0, x), 511)
                y = min(max(0, y), 511)
                
                # 在对应位置标记一个点，颜色表示密度
                plot_image[y, x, 1] = min(plot_image[y, x, 1] + 50, 255)  # 绿色通道累加
        
        # 应用高斯模糊使点更明显
        plot_image = cv.GaussianBlur(plot_image, (5, 5), 0)
        
        # 绘制坐标轴
        # X轴（水平相位）
        cv.line(plot_image, (0, 511), (511, 511), (0, 0, 255), 2)
        cv.putText(plot_image, "X相位", (400, 500), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Y轴（垂直相位）
        cv.line(plot_image, (0, 511), (0, 0), (255, 0, 0), 2)
        cv.putText(plot_image, "Y相位", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 保存3D相位分布图（顶视图）
        cv.imwrite("results/phase_3d_distribution.png", plot_image)
        
        # 选择最合适的可视化结果作为返回值
        combined_visual = phase_dist_map  # 使用二维相位分布图作为主要可视化结果
        
        print("\n水平和垂直相位组合完成:")
        print(f"  水平相位范围: [0, {h_max:.2f}]")
        print(f"  垂直相位范围: [0, {v_max:.2f}]")
        print("  已创建多种相位组合可视化结果:")
        print("    1. 二维相位分布图 (results/phase_distribution_map.png)")
        print("    2. 相位等值线图 (results/phase_contour_map.png)")
        print("    3. 3D相位分布图 (results/phase_3d_distribution.png)")
        print("    4. 单独的水平和垂直相位图")
        print("  原始相位数据已保存为NPY格式，可用于后续精确处理")
        
        return combined_visual, combined_data

    def get_k1_k2(self):
        """
        获得k1和k2矩阵
        
        该方法通过解码二值化格雷码图像，计算用于相位解包裹的k1和k2矩阵。
        k1基于4位格雷码，k2基于5位格雷码。这两个矩阵用于确定每个像素点
        所处的条纹周期，是解包裹相位的关键。
        
        返回:
            tuple: (k1矩阵, k2矩阵)，分别表示每个像素点所处的周期编号
        """
        # 获取二值化格雷码图像
        BCG = self.getBinarizedGrayCodes()
        rows, cols = BCG[0].shape
        
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
                code_k1 = code1 + str(BCG[0][a,b]) + str(BCG[1][a,b]) + str(BCG[2][a,b]) + str(BCG[3][a,b])
                
                # 组合前5位格雷码，用于计算k2
                code_k2 = code1 + str(BCG[0][a,b]) + str(BCG[1][a,b]) + str(BCG[2][a,b]) + str(BCG[3][a,b]) + str(BCG[4][a,b])
                
                try:
                    # 查询格雷码映射表，获得对应的十进制值
                    k1[a,b] = g_k1.code2k[code_k1]  # 4位格雷码对应的十进制数
                    k2[a,b] = g_k2.code2k[code_k2]  # 修改：直接使用解码值，不再应用修正公式
                except KeyError as e:
                    print(f"错误: 在位置({a},{b})处的格雷码无法解码: {e}")
                    print(f"code_k1: {code_k1}, code_k2: {code_k2}")
                    print(f"BCG[0-4]的值: {BCG[0][a,b]}, {BCG[1][a,b]}, {BCG[2][a,b]}, {BCG[3][a,b]}, {BCG[4][a,b]}")
                    
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


# 如果直接运行此文件
if __name__ == "__main__":
    print("解包裹相位计算程序 - 主程序")
    print("==================================")
    
    # 用户选项
    print("\n显示选项：")
    print("1. 不显示中间过程图像（只显示最终结果）")
    print("2. 显示所有过程图像（包括跳变检测、质量评估等）")
    
    try:
        # 获取用户输入的显示选项
        choice = input("请选择显示选项 (1/2)，默认为1: ")
        if choice not in ["1", "2", ""]:
            print("无效选择，使用默认选项1")
            choice = "1"
        if choice == "":
            choice = "1"
            
        show_details = choice == "2"
        
        # 方向选择
        print("\n解包裹方向选项：")
        print("1. 仅水平方向解包裹")
        print("2. 仅垂直方向解包裹")
        print("3. 水平和垂直方向解包裹并组合 (默认)")
        
        direction_choice = input("请选择解包裹方向 (1/2/3)，默认为3: ")
        if direction_choice not in ["1", "2", "3", ""]:
            print("无效选择，使用默认选项3")
            direction_choice = "3"
        if direction_choice == "":
            direction_choice = "3"
            
        # 设置解包裹方向
        if direction_choice == "1":
            unwrap_direction = "horizontal"
            print("将仅进行水平方向解包裹")
        elif direction_choice == "2":
            unwrap_direction = "vertical"
            print("将仅进行垂直方向解包裹")
        else:
            unwrap_direction = "both"
            print("将进行水平和垂直方向解包裹并组合结果")
        
        # 图像尺寸标准化选项
        print("\n图像尺寸标准化选项：")
        print("1. 自动裁剪到最小尺寸 (默认)")
        print("2. 缩放所有图像到相同尺寸")
        
        size_choice = input("请选择图像尺寸标准化方法 (1/2)，默认为1: ")
        size_method = "crop" if size_choice != "2" else "resize"
        
        # 读取包裹相位图像
        wrapped_phase_h_path = None
        wrapped_phase_v_path = None
        
        if unwrap_direction in ["horizontal", "both"]:
            wrapped_phase_h_path = input("\n请输入水平方向包裹相位图像的路径: ")
            if not os.path.exists(wrapped_phase_h_path):
                print(f"错误: 文件 {wrapped_phase_h_path} 不存在")
                exit()
        
        if unwrap_direction in ["vertical", "both"]:
            wrapped_phase_v_path = input("\n请输入垂直方向包裹相位图像的路径: ")
            if not os.path.exists(wrapped_phase_v_path):
                print(f"错误: 文件 {wrapped_phase_v_path} 不存在")
                exit()
        
        # 读取格雷码图像
        print("\n需要读取格雷码图像")
        
        # 水平方向解包裹需要垂直条纹格雷码图像
        graycode_h_paths = []
        if unwrap_direction in ["horizontal", "both"]:
            print("\n请输入用于水平方向解包裹的5张垂直条纹格雷码图像路径:")
            for i in range(5):
                path = input(f"垂直条纹格雷码图像 {i+1}/5: ")
                if not os.path.exists(path):
                    print(f"错误: 文件 {path} 不存在")
                    exit()
                graycode_h_paths.append(path)
        
        # 垂直方向解包裹需要水平条纹格雷码图像
        graycode_v_paths = []
        if unwrap_direction in ["vertical", "both"]:
            print("\n请输入用于垂直方向解包裹的5张水平条纹格雷码图像路径:")
            for i in range(5):
                path = input(f"水平条纹格雷码图像 {i+1}/5: ")
                if not os.path.exists(path):
                    print(f"错误: 文件 {path} 不存在")
                    exit()
                graycode_v_paths.append(path)
        
        # 是否使用自适应阈值
        adaptive_choice = input("\n是否使用自适应阈值进行二值化? (y/n)，默认为n: ")
        adaptive_threshold = adaptive_choice.lower() == "y"
        
        # 创建相位解包裹器
        unwrapper = PhaseUnwrapper(n=5, direction=unwrap_direction)
        
        # 执行水平方向解包裹
        horizontal_phase = None
        if unwrap_direction in ["horizontal", "both"]:
            print("\n===== 执行水平方向解包裹 =====")
            
            # 读取包裹相位图像
            wrapped_phase_h = cv.imread(wrapped_phase_h_path, -1)
            if wrapped_phase_h is None:
                print(f"错误: 无法读取包裹相位图像 {wrapped_phase_h_path}")
                exit()
            
            # 转换为float32类型并确保范围在[0, 2π]
            if wrapped_phase_h.dtype != np.float32:
                if wrapped_phase_h.dtype == np.uint8:
                    # 假设图像为[0, 255]范围，转换为[0, 2π]
                    wrapped_phase_h = wrapped_phase_h.astype(np.float32) * (2 * math.pi / 255)
                elif wrapped_phase_h.dtype == np.uint16:
                    # 假设图像为[0, 65535]范围，转换为[0, 2π]
                    wrapped_phase_h = wrapped_phase_h.astype(np.float32) * (2 * math.pi / 65535)
                else:
                    # 默认情况，假设已经在[0, 2π]范围内
                    wrapped_phase_h = wrapped_phase_h.astype(np.float32)
            
            # 读取格雷码图像
            graycode_h_images = []
            for path in graycode_h_paths:
                img = cv.imread(path, -1)
                if img is None:
                    print(f"错误: 无法读取格雷码图像 {path}")
                    exit()
                graycode_h_images.append(img)
            
            # 执行解包裹
            unwrapper.direction = "horizontal"
            horizontal_phase = unwrapper.unwrap_phase(
                wrapped_phase_h,
                graycode_h_images,
                adaptive_threshold=adaptive_threshold,
                show_results=show_details
            )
        
        # 执行垂直方向解包裹
        vertical_phase = None
        if unwrap_direction in ["vertical", "both"]:
            print("\n===== 执行垂直方向解包裹 =====")
            
            # 读取包裹相位图像
            wrapped_phase_v = cv.imread(wrapped_phase_v_path, -1)
            if wrapped_phase_v is None:
                print(f"错误: 无法读取包裹相位图像 {wrapped_phase_v_path}")
                exit()
            
            # 转换为float32类型并确保范围在[0, 2π]
            if wrapped_phase_v.dtype != np.float32:
                if wrapped_phase_v.dtype == np.uint8:
                    # 假设图像为[0, 255]范围，转换为[0, 2π]
                    wrapped_phase_v = wrapped_phase_v.astype(np.float32) * (2 * math.pi / 255)
                elif wrapped_phase_v.dtype == np.uint16:
                    # 假设图像为[0, 65535]范围，转换为[0, 2π]
                    wrapped_phase_v = wrapped_phase_v.astype(np.float32) * (2 * math.pi / 65535)
                else:
                    # 默认情况，假设已经在[0, 2π]范围内
                    wrapped_phase_v = wrapped_phase_v.astype(np.float32)
            
            # 读取格雷码图像
            graycode_v_images = []
            for path in graycode_v_paths:
                img = cv.imread(path, -1)
                if img is None:
                    print(f"错误: 无法读取格雷码图像 {path}")
                    exit()
                graycode_v_images.append(img)
            
            # 执行解包裹
            unwrapper.direction = "vertical"
            vertical_phase = unwrapper.unwrap_phase(
                wrapped_phase_v,
                graycode_v_images,
                adaptive_threshold=adaptive_threshold,
                show_results=show_details
            )
        
        # 组合水平和垂直方向的解包裹相位
        if unwrap_direction == "both" and horizontal_phase is not None and vertical_phase is not None:
            print("\n===== 组合水平和垂直方向解包裹相位 =====")
            
            # 确保两个相位矩阵的尺寸一致
            if horizontal_phase.shape != vertical_phase.shape:
                print("警告: 水平和垂直相位矩阵尺寸不一致，将调整到相同尺寸")
                min_rows = min(horizontal_phase.shape[0], vertical_phase.shape[0])
                min_cols = min(horizontal_phase.shape[1], vertical_phase.shape[1])
                
                horizontal_phase = horizontal_phase[:min_rows, :min_cols]
                vertical_phase = vertical_phase[:min_rows, :min_cols]
            
            # 创建双通道数据结构，同时保存水平和垂直方向的相位值
            combined_data = np.zeros((horizontal_phase.shape[0], horizontal_phase.shape[1], 2), dtype=np.float32)
            combined_data[:,:,0] = horizontal_phase  # 第一个通道存储水平方向的相位
            combined_data[:,:,1] = vertical_phase    # 第二个通道存储垂直方向的相位
            
            # 创建可视化结果目录
            if not os.path.exists("results"):
                os.makedirs("results")
                
            # 使用numpy的格式保存原始相位数据
            np.save("results/combined_phase_data.npy", combined_data)
            np.save("results/horizontal_phase.npy", horizontal_phase)
            np.save("results/vertical_phase.npy", vertical_phase)
            
            # 归一化相位值
            h_max = np.max(horizontal_phase)
            v_max = np.max(vertical_phase)
            
            h_norm = horizontal_phase / h_max if h_max > 0 else horizontal_phase
            v_norm = vertical_phase / v_max if v_max > 0 else vertical_phase
            
            # 创建伪彩色图
            h_scaled = (h_norm * 255).astype(np.uint8)
            v_scaled = (v_norm * 255).astype(np.uint8)
            
            h_color = cv.applyColorMap(h_scaled, cv.COLORMAP_JET)
            v_color = cv.applyColorMap(v_scaled, cv.COLORMAP_JET)
            
            # 保存单独的相位图
            cv.imwrite("results/horizontal_phase_color.png", h_color)
            cv.imwrite("results/vertical_phase_color.png", v_color)
            
            # 创建二维相位分布图
            # 将水平和垂直相位归一化到[0,1]范围，然后使用HSV颜色空间表示
            # 色调(H)用于表示相位角度(atan2(v_norm, h_norm))
            # 饱和度(S)保持最大
            # 亮度(V)表示相位幅值(sqrt(h_norm^2 + v_norm^2))
            
            # 计算相位角度和幅值
            phase_angle = np.arctan2(v_norm, h_norm) # 范围[-pi, pi]
            phase_angle = (phase_angle + np.pi) / (2 * np.pi) # 归一化到[0,1]
            
            phase_magnitude = np.sqrt(h_norm**2 + v_norm**2)
            # 归一化幅值到[0,1]
            phase_magnitude = np.clip(phase_magnitude, 0, 1)
            
            # 创建HSV图像
            hsv_image = np.zeros((horizontal_phase.shape[0], horizontal_phase.shape[1], 3), dtype=np.float32)
            hsv_image[:,:,0] = phase_angle * 179 # H通道，范围[0,179]
            hsv_image[:,:,1] = 255              # S通道，最大饱和度
            hsv_image[:,:,2] = phase_magnitude * 255 # V通道，范围[0,255]
            
            # 转换HSV到BGR
            hsv_image = hsv_image.astype(np.uint8)
            phase_dist_map = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)
            
            # 保存二维相位分布图
            cv.imwrite("results/phase_distribution_map.png", phase_dist_map)
            
            # 显示组合结果
            cv.imshow("水平方向相位图", h_color)
            cv.imshow("垂直方向相位图", v_color)
            cv.imshow("二维相位分布图", phase_dist_map)
            
            # 等待用户按键
            print("\n交互提示:")
            print("- 按's'键保存图像")
            print("- 按'q'键或ESC键退出")
            
            while True:
                key = cv.waitKey(0)
                if key == ord("s"):
                    # 按's'键保存图像
                    cv.imwrite("results/combined_phase_user.png", phase_dist_map)
                    print("已保存组合相位图")
                elif key == ord("q") or key == 27:  # 'q'键或ESC键
                    break
            
            cv.destroyAllWindows()
        
        print("\n相位解包裹处理完成!")
        
    except Exception as e:
        print(f"程序执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
        cv.destroyAllWindows()
        print("\n程序异常终止")