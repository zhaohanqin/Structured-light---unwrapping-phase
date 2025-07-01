import numpy as np
import cv2 as cv
import math
from wrapped_phase_algorithm import WrappedPhase
from graycode_binarization import Binariization
from generate_graycode_map import GrayCode

class UnwrappedPhase():
    """
    解包裹相位计算类
    
    该类实现了基于格雷码辅助的相位解包裹算法。
    解包裹是将包裹相位（范围[0, 2π]）转换为连续绝对相位的过程。
    
    算法原理：
    1. 首先通过四步相移法获得包裹相位
    2. 使用格雷码图案确定每个像素点所在的条纹周期
    3. 根据包裹相位的区间选择解包裹策略
    4. 计算连续的绝对相位
    
    格雷码辅助解包裹的优势：
    - 提高解包裹的鲁棒性
    - 减少相位跳跃错误
    - 适用于复杂表面和噪声环境
    """
    
    def __init__(self, n: int = 5):
        """
        初始化解包裹相位计算器
        
        参数:
            n (int): 格雷码位数，默认为5
                     5位格雷码可以编码32个不同的条纹周期
                     位数越多，编码范围越大，但计算复杂度也越高
        """
        self.n = n
    
    def getBinarizedGrayCodes(self, m: int = 5):
        """
        获得二值化后的格雷码图像
        
        该方法读取已经二值化处理的格雷码图像。
        二值化是将格雷码图像转换为黑白二值图像的过程，
        便于后续的格雷码解码。
        
        参数:
            m (int): 格雷码位数，默认为5
                     应该与初始化时的n值一致
                     5位格雷码对应5幅二值化图像
        
        返回:
            list: 包含m个numpy数组的列表，每个数组代表一幅二值化格雷码图像
                  图像数据类型为uint8，像素值为0或1
                  尺寸：(height, width)
        
        注意:
            - 图像文件名格式为"binarized_GC-0.png", "binarized_GC-1.png"等
            - 图像应该已经过二值化处理，像素值为0（黑）或255（白）
            - 函数会将像素值归一化到[0,1]范围
        """
        BGC = []
        for i in range(self.n):
            # 使用新生成的匹配格雷码图像
            filename = r"gray_patterns\matched_binary_" + str(i) + ".png"
            
            # 读取图像，flags=0表示以灰度图像方式读取
            img = np.array(cv.imread(filename, 0), np.uint8)
            
            # 确保图像成功读取
            if img is None:
                raise FileNotFoundError(f"无法读取图像文件: {filename}")
                
            # 将像素值从[0,255]归一化到[0,1]范围
            # 二值化图像中，0表示黑，255表示白
            img_scaled = img/255
            
            # 转换为uint8类型，此时像素值为0或1
            BGC.append(img_scaled.astype(np.uint8))
            
            # 调试: 显示读取的图像
            if __name__ == "__main__":
                cv.imshow(f"Binary Gray Code {i}", img)
                cv.waitKey(500)  # 显示500ms
        
        if __name__ == "__main__":
            cv.destroyAllWindows()
            
        return BGC

    def get_k1_k2(self):
        """
        获得k1和k2矩阵
        
        该方法通过解码格雷码图像获得k1和k2矩阵。
        k1和k2分别表示不同精度下的条纹周期数，
        用于后续的相位解包裹计算。
        
        算法步骤：
        1. 读取二值化格雷码图像
        2. 对每个像素点，组合前4位或前5位格雷码
        3. 查询格雷码映射表，获得对应的十进制值
        4. 计算k1和k2矩阵
        
        返回:
            tuple: (k1, k2) 两个numpy数组
                   k1: 4位格雷码解码结果，数据类型uint8，编码范围[0, 15]
                   k2: 5位格雷码解码结果，数据类型uint8，编码范围[0, 31]
                   尺寸：(height, width)
        
        算法说明：
        - k1使用前4位格雷码，编码范围[0, 15]
        - k2使用前5位格雷码，编码范围[0, 31]
        - k2的计算公式：k2 = floor((code2k[code_k2] + 1) / 2)
          这样可以获得更精确的条纹周期估计
        """
        # 获取二值化格雷码图像
        BCG = self.getBinarizedGrayCodes()
        rows, cols = BCG[0].shape
        
        # 调试: 打印二值化格雷码图像的尺寸
        print(f"二值化格雷码图像尺寸: {rows}x{cols}")
        
        # 初始化k1和k2矩阵
        k1 = np.zeros((rows, cols), np.uint8)  # 4位格雷码解码结果
        k2 = np.zeros((rows, cols), np.uint8)  # 5位格雷码解码结果
        
        # 创建格雷码映射对象
        g_k1 = GrayCode(4)  # 4位格雷码映射器
        g_k2 = GrayCode(5)  # 5位格雷码映射器
        
        # 调试：打印格雷码映射表的大小
        print(f"4位格雷码映射表大小: {len(g_k1.code2k)}")
        print(f"5位格雷码映射表大小: {len(g_k2.code2k)}")
        
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
                    
                    # 计算k2：使用5位格雷码，并应用修正公式
                    # 公式：k2 = floor((code2k[code_k2] + 1) / 2)
                    # 这样可以获得更精确的条纹周期估计
                    k2[a,b] = g_k2.code2k[code_k2]  # 修改：直接使用解码值，不再应用修正公式
                except KeyError as e:
                    print(f"错误: 在位置({a},{b})处的格雷码无法解码: {e}")
                    print(f"code_k1: {code_k1}, code_k2: {code_k2}")
                    print(f"BCG[0-4]的值: {BCG[0][a,b]}, {BCG[1][a,b]}, {BCG[2][a,b]}, {BCG[3][a,b]}, {BCG[4][a,b]}")
                    raise
        
        # 调试: 打印部分k1和k2的值
        print("\n部分k1值:")
        print(k1[rows//2, cols//4:cols//4+10])  # 打印中间行的一部分
        
        print("\n部分k2值:")
        print(k2[rows//2, cols//4:cols//4+10])  # 打印中间行的一部分
        
        # 检查k1和k2的范围是否合理
        print(f"\nk1的范围: [{np.min(k1)}, {np.max(k1)}], 理论范围: [0, 15]")
        print(f"k2的范围: [{np.min(k2)}, {np.max(k2)}], 理论范围: [0, 31]")
        
        # 打印k2计算的详细信息
        sample_row = rows//2
        sample_col = cols//4
        print(f"\nk2计算示例 (位置 {sample_row},{sample_col}):")
        code_k2_sample = str(BCG[0][sample_row,sample_col]) + str(BCG[1][sample_row,sample_col]) + str(BCG[2][sample_row,sample_col]) + str(BCG[3][sample_row,sample_col]) + str(BCG[4][sample_row,sample_col])
        print(f"格雷码: {code_k2_sample}")
        print(f"解码值: {g_k2.code2k[code_k2_sample]}")
        
        # 可视化k1和k2
        if __name__ == "__main__":
            # 将k1和k2缩放到[0,255]以便显示
            k1_vis = (k1 * (255/15)).astype(np.uint8)
            k2_vis = (k2 * (255/31)).astype(np.uint8)
            
            cv.imshow("k1 matrix", k1_vis)
            cv.imshow("k2 matrix", k2_vis)
            cv.waitKey(1000)  # 显示1秒
            cv.destroyAllWindows()
        
        return k1, k2

    def computeUnwrappedPhase(self):
        """
        计算解包裹相位
        
        该方法实现了格雷码辅助的相位解包裹算法。
        通过结合包裹相位和格雷码信息，计算连续的绝对相位。
        
        算法步骤：
        1. 调用包裹相位计算模块，获得包裹相位
        2. 调用格雷码解码模块，获得k1和k2矩阵
        3. 根据包裹相位的区间选择解包裹策略
        4. 计算连续的绝对相位
        
        返回:
            numpy.ndarray: 解包裹后的绝对相位矩阵
                          数据类型为float16，节省内存
                          相位值范围：[0, 32π]（对于5位格雷码）
                          尺寸：(height, width)
        
        解包裹策略：
        - 当包裹相位 ≤ π/2 时：
          φ_unwrapped = φ_wrapped + k2*2π
          使用k2（5位格雷码）进行解包裹
        
        - 当 π/2 < 包裹相位 < 3π/2 时：
          φ_unwrapped = φ_wrapped + k1*2π
          使用k1（4位格雷码）进行解包裹
        
        - 当包裹相位 ≥ 3π/2 时：
          φ_unwrapped = φ_wrapped + (k2-1)*2π
          使用修正后的k2进行解包裹
        
        算法优势：
        - 结合了不同精度的格雷码信息
        - 根据相位区间选择最优的解包裹策略
        - 提高了相位解包裹的准确性和鲁棒性
        """
        # 创建包裹相位计算器实例
        WP = WrappedPhase()
        
        # 计算包裹相位
        wrapped_pha = WP.computeWrappedphase(WP.getImageData())
        
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
        
        # 获取图像尺寸
        rows, cols = k1.shape
        
        # 初始化解包裹相位矩阵
        # 使用float16类型以节省内存，对于相位计算精度足够
        unwrapped_pha = np.zeros((rows, cols), np.float16)
        
        # 逐像素进行相位解包裹
        for c in range(rows):
            for d in range(cols):
                if wrapped_pha[c,d] <= math.pi/2:
                    # 第一区间：使用k2进行解包裹
                    # 当包裹相位较小时，使用更高精度的k2
                    unwrapped_pha[c,d] = wrapped_pha[c,d] + k2[c,d]*2*math.pi
                    
                elif wrapped_pha[c, d] > math.pi / 2 and wrapped_pha[c, d] < 3*math.pi / 2:
                    # 第二区间：使用k1进行解包裹
                    # 当包裹相位在中间范围时，使用k1
                    unwrapped_pha[c, d] = wrapped_pha[c, d] + k1[c, d] * 2 * math.pi
                    
                elif wrapped_pha[c, d] >= 3*math.pi / 2:
                    # 第三区间：使用修正后的k2进行解包裹
                    # 当包裹相位较大时，使用修正后的k2
                    unwrapped_pha[c, d] = wrapped_pha[c, d] + (k2[c, d]-1) * 2 * math.pi
        
        # 调试: 打印解包裹相位的范围
        print(f"\n解包裹相位的范围: [{np.min(unwrapped_pha)}, {np.max(unwrapped_pha)}]")
        print(f"理论范围: [0, {32*math.pi}]")
        
        return unwrapped_pha

    def showUnwrappedPhase(self):
        """
        显示解包裹相位
        
        该方法将解包裹相位可视化显示，并提供保存功能。
        由于绝对相位的值范围较大，需要进行适当的缩放才能显示。
        
        显示步骤：
        1. 计算解包裹相位
        2. 将相位值缩放到[0,255]范围
        3. 显示图像
        4. 提供保存选项
        
        缩放说明：
        - 原始相位范围：[0, 32π]（对于5位格雷码）
        - 缩放公式：phase_scaled = round(phase * 255 / (32*π))
        - 缩放后范围：[0, 255]，适合图像显示
        
        交互功能：
        - 按任意键关闭图像窗口
        - 按's'键保存图像为"Absolute_pha.png"
        - 鼠标点击图像可显示该点的相位值
        """
        # 计算解包裹相位（使用改进的算法）
        upha = self.computeUnwrappedPhase()
        
        # 将相位值缩放到[0,255]范围用于显示
        # 对于5位格雷码，相位范围约为[0, 32π]
        # 缩放公式：phase_scaled = round(phase * 255 / (32*π))
        upha_scaled = np.rint(upha*255/(32*math.pi))
        
        # 转换为uint8类型用于图像显示
        upha_scaled_uint = upha_scaled.astype(np.uint8)
        
        # 创建results目录（如果不存在）
        import os
        if not os.path.exists("results"):
            os.makedirs("results")
        
        # 保存不同视图的解包裹相位图像
        # 1. 原始缩放视图
        cv.imwrite("results/unwrapped_phase_original.png", upha_scaled_uint)
        
        # 2. 应用伪彩色映射以增强可视化效果
        upha_color = cv.applyColorMap(upha_scaled_uint, cv.COLORMAP_JET)
        cv.imwrite("results/unwrapped_phase_color.png", upha_color)
        
        # 3. 应用直方图均衡化以增强对比度
        upha_eq = cv.equalizeHist(upha_scaled_uint)
        cv.imwrite("results/unwrapped_phase_equalized.png", upha_eq)
        
        # 4. 3D可视化（保存为高度图）
        # 将相位值归一化到[0,1]范围
        upha_norm = upha / np.max(upha)
        # 保存为16位PNG，以保留更多细节
        cv.imwrite("results/unwrapped_phase_height.png", (upha_norm * 65535).astype(np.uint16))
        
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
                cv.imshow("Unwrapped Phase (Color)", display_img)
        
        # 显示解包裹相位图像
        cv.imshow("Unwrapped Phase", upha_scaled_uint)
        cv.imshow("Unwrapped Phase (Color)", upha_color)
        cv.imshow("Unwrapped Phase (Equalized)", upha_eq)
        
        # 设置鼠标回调
        cv.setMouseCallback("Unwrapped Phase (Color)", mouse_callback)
        
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
                cv.imwrite("results/Absolute_pha.png", upha_scaled_uint)
                print("已保存图像: results/Absolute_pha.png")
            elif key == ord("q") or key == 27:  # 'q'键或ESC键
                break
        
        # 关闭所有图像窗口
        cv.destroyAllWindows()
        
        print("解包裹相位图像已保存到results目录")
        return upha

    def verify_graycodes(self):
        """验证格雷码图像与解码映射表是否一致"""
        # 读取二值化格雷码图像
        BCG = self.getBinarizedGrayCodes()
        
        # 创建格雷码映射对象
        g = GrayCode(5)
        
        # 生成理想的格雷码图案
        ideal_patterns = []
        for i in range(5):
            pattern = g.toPattern(i, cols=BCG[0].shape[1], rows=BCG[0].shape[0])
            _, ideal = cv.threshold(pattern, 127, 1, cv.THRESH_BINARY)
            ideal_patterns.append(ideal.astype(np.uint8))
        
        # 比较实际图像与理想图像
        for i in range(5):
            diff = cv.absdiff(BCG[i], ideal_patterns[i])
            error_rate = np.sum(diff) / (BCG[i].shape[0] * BCG[i].shape[1])
            print(f"格雷码图像{i}的误差率: {error_rate*100:.2f}%")
            
            # 保存差异图像而不是显示
            cv.imwrite(f"gray_patterns/diff_{i}.png", diff*255)
            
            # 保存理想格雷码图像
            cv.imwrite(f"gray_patterns/ideal_{i}.png", ideal_patterns[i]*255)
            
        print("差异图像和理想格雷码图像已保存到gray_patterns目录")
        return

    def analyze_unwrapped_phase(self, unwrapped_phase):
        """
        分析解包裹相位的质量和改进效果
        
        参数:
            unwrapped_phase: 解包裹相位矩阵
            
        返回:
            None
        """
        print("\n===== 解包裹相位质量分析 =====")
        
        # 1. 相位范围分析
        phase_min = np.min(unwrapped_phase)
        phase_max = np.max(unwrapped_phase)
        phase_mean = np.mean(unwrapped_phase)
        phase_std = np.std(unwrapped_phase)
        
        print(f"相位范围: [{phase_min}, {phase_max}]")
        print(f"相位均值: {phase_mean}")
        print(f"相位标准差: {phase_std}")
        
        # 2. 相位连续性分析
        # 计算水平和垂直方向的相位梯度
        grad_x = np.abs(np.diff(unwrapped_phase, axis=1))
        grad_y = np.abs(np.diff(unwrapped_phase, axis=0))
        
        # 计算大梯度点的数量（可能的相位跳变）
        threshold = 0.5 * math.pi  # 相位跳变阈值
        jumps_x = np.sum(grad_x > threshold)
        jumps_y = np.sum(grad_y > threshold)
        total_pixels = unwrapped_phase.size
        
        print(f"水平方向相位跳变点数量: {jumps_x} ({jumps_x/total_pixels*100:.4f}%)")
        print(f"垂直方向相位跳变点数量: {jumps_y} ({jumps_y/total_pixels*100:.4f}%)")
        
        # 3. 相位分布分析
        # 计算相位值的直方图
        hist, bins = np.histogram(unwrapped_phase, bins=32)
        
        # 打印直方图信息
        print("\n相位分布直方图:")
        for i in range(len(hist)):
            if i % 4 == 0:  # 每4个bin打印一次，减少输出量
                bin_start = bins[i]
                bin_end = bins[i+1]
                count = hist[i]
                percentage = count / total_pixels * 100
                print(f"  [{bin_start:.2f}, {bin_end:.2f}): {count} 像素 ({percentage:.2f}%)")
        
        # 4. 保存分析结果
        with open("results/phase_analysis.txt", "w") as f:
            f.write("===== 解包裹相位质量分析 =====\n\n")
            f.write(f"相位范围: [{phase_min}, {phase_max}]\n")
            f.write(f"相位均值: {phase_mean}\n")
            f.write(f"相位标准差: {phase_std}\n\n")
            
            f.write(f"水平方向相位跳变点数量: {jumps_x} ({jumps_x/total_pixels*100:.4f}%)\n")
            f.write(f"垂直方向相位跳变点数量: {jumps_y} ({jumps_y/total_pixels*100:.4f}%)\n\n")
            
            f.write("相位分布直方图:\n")
            for i in range(len(hist)):
                bin_start = bins[i]
                bin_end = bins[i+1]
                count = hist[i]
                percentage = count / total_pixels * 100
                f.write(f"  [{bin_start:.2f}, {bin_end:.2f}): {count} 像素 ({percentage:.2f}%)\n")
        
        print("\n分析结果已保存到 results/phase_analysis.txt")

    def phase_unwrapping_with_continuity(self, wrapped_pha, k1, k2):
        """
        添加相位连续性约束的解包裹算法
        
        该方法通过考虑相邻像素之间的关系，确保相位的连续性，
        减少相位跳变问题。
        
        参数:
            wrapped_pha: 包裹相位矩阵
            k1: 4位格雷码解码结果
            k2: 5位格雷码解码结果
            
        返回:
            numpy.ndarray: 考虑连续性约束的解包裹相位矩阵
        """
        rows, cols = wrapped_pha.shape
        unwrapped_pha = np.zeros_like(wrapped_pha, dtype=np.float32)
        
        # 先处理第一个像素
        if wrapped_pha[0, 0] <= math.pi/2:
            unwrapped_pha[0, 0] = wrapped_pha[0, 0] + k2[0, 0]*2*math.pi
        elif wrapped_pha[0, 0] < 3*math.pi/2:
            unwrapped_pha[0, 0] = wrapped_pha[0, 0] + k1[0, 0]*2*math.pi
        else:
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
        
        该方法使用中值滤波去除离群值，然后使用高斯滤波进一步平滑，
        减少相位噪声和跳变。
        
        参数:
            unwrapped_pha: 解包裹相位矩阵
            kernel_size: 滤波核大小，默认为5
            
        返回:
            numpy.ndarray: 平滑处理后的相位矩阵
        """
        # 将相位矩阵转换为float32类型，确保滤波操作的精度
        phase_float32 = unwrapped_pha.astype(np.float32)
        
        # 使用中值滤波去除离群值（相位跳变点）
        # 中值滤波对于去除椒盐噪声（如相位跳变）非常有效
        median_filtered = cv.medianBlur(phase_float32, kernel_size)
        
        # 使用高斯滤波进一步平滑
        # 高斯滤波可以保留相位的整体结构，同时减少小的波动
        smoothed = cv.GaussianBlur(median_filtered, (kernel_size, kernel_size), 0)
        
        # 可视化平滑前后的差异
        if __name__ == "__main__":
            # 计算平滑前后的差异
            diff = np.abs(unwrapped_pha - smoothed)
            
            # 将差异缩放到[0,255]范围用于显示
            diff_scaled = (diff * 255 / np.max(diff)).astype(np.uint8)
            
            # 应用伪彩色映射以增强可视化效果
            diff_color = cv.applyColorMap(diff_scaled, cv.COLORMAP_JET)
            
            # 保存差异图像
            cv.imwrite("results/smoothing_difference.png", diff_color)
            
            # 可视化平滑后的相位
            smoothed_scaled = (smoothed * 255 / np.max(smoothed)).astype(np.uint8)
            smoothed_color = cv.applyColorMap(smoothed_scaled, cv.COLORMAP_JET)
            cv.imwrite("results/smoothed_phase.png", smoothed_color)
            
            print(f"平滑处理前后的最大差异: {np.max(diff):.6f} rad")
            print(f"平滑处理前后的平均差异: {np.mean(diff):.6f} rad")
        
        return smoothed

    def estimate_phase_quality(self, fringe_images):
        """
        估计相位质量
        
        该方法使用条纹图像的调制度作为相位质量的指标。
        调制度越高，相位质量越好；调制度越低，相位质量越差。
        
        参数:
            fringe_images: 四步相移的条纹图像列表 [I0, I1, I2, I3]
            
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
        
        # 可视化相位质量
        if __name__ == "__main__":
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
        
        return modulation

    def visualize_phase_jumps(self, unwrapped_pha, threshold=0.5):
        """
        可视化相位跳变区域
        
        该方法检测并可视化相位跳变区域，帮助分析解包裹算法的质量。
        
        参数:
            unwrapped_pha: 解包裹相位矩阵
            threshold: 相位跳变阈值，默认为0.5（相位梯度大于0.5*pi被认为是跳变）
            
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
        
        # 可视化相位跳变区域
        if __name__ == "__main__":
            # 保存二值化跳变图像
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
            
            # 分析跳变区域的分布
            # 计算每行和每列的跳变点数量
            jumps_per_row = np.sum(jumps > 0, axis=1)
            jumps_per_col = np.sum(jumps > 0, axis=0)
            
            # 找出跳变最多的行和列
            max_jump_row = np.argmax(jumps_per_row)
            max_jump_col = np.argmax(jumps_per_col)
            
            print(f"  跳变最多的行: {max_jump_row}, 跳变点数: {jumps_per_row[max_jump_row]}")
            print(f"  跳变最多的列: {max_jump_col}, 跳变点数: {jumps_per_col[max_jump_col]}")
        
        return jumps

    def optimize_phase_jumps(self, unwrapped_pha, jumps, quality):
        """
        进一步优化相位跳变区域
        
        该方法使用加权平滑处理来优化相位跳变区域，
        根据相位质量和跳变区域进行自适应处理。
        
        参数:
            unwrapped_pha: 解包裹相位矩阵
            jumps: 相位跳变区域的二值图像
            quality: 相位质量矩阵
            
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
        
        # 可视化优化效果
        if __name__ == "__main__":
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
            optimized_jumps = self.visualize_phase_jumps(optimized_pha, threshold=0.5)
            
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
        
        return optimized_pha

if __name__ == "__main__":
    # 测试代码：创建解包裹相位计算器实例并显示结果
    u = UnwrappedPhase()
    
    # 验证格雷码图像与解码映射表的一致性
    print("正在验证格雷码图像与解码映射表的一致性...")
    u.verify_graycodes()
    
    # 修复解包裹相位算法中的边界条件
    def improved_computeUnwrappedPhase(self):
        """改进的解包裹相位计算方法，整合所有改进功能"""
        # 创建包裹相位计算器实例
        WP = WrappedPhase()
        
        # 获取四步相移图像
        fringe_images = WP.getImageData()
        
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
        
        # 获取图像尺寸
        rows, cols = k1.shape
        
        # 可视化包裹相位和各区域
        if __name__ == "__main__":
            # 显示包裹相位
            wrapped_vis = (wrapped_pha * 255 / (2*math.pi)).astype(np.uint8)
            cv.imshow("Wrapped Phase", wrapped_vis)
            cv.waitKey(1000)
            
            # 第一区间：wrapped_pha <= pi/2
            region1 = np.zeros_like(wrapped_pha)
            region1[wrapped_pha <= math.pi/2] = 255
            cv.imshow("Region 1 (wrapped_pha <= pi/2)", region1.astype(np.uint8))
            cv.waitKey(1000)
            
            # 第二区间：pi/2 < wrapped_pha < 3pi/2
            region2 = np.zeros_like(wrapped_pha)
            region2[(wrapped_pha > math.pi/2) & (wrapped_pha < 3*math.pi/2)] = 255
            cv.imshow("Region 2 (pi/2 < wrapped_pha < 3pi/2)", region2.astype(np.uint8))
            cv.waitKey(1000)
            
            # 第三区间：wrapped_pha >= 3pi/2
            region3 = np.zeros_like(wrapped_pha)
            region3[wrapped_pha >= 3*math.pi/2] = 255
            cv.imshow("Region 3 (wrapped_pha >= 3pi/2)", region3.astype(np.uint8))
            cv.waitKey(1000)
            
            cv.destroyAllWindows()
        
        # 1. 使用相位连续性约束的解包裹算法
        print("\n步骤1: 使用相位连续性约束的解包裹算法...")
        unwrapped_pha = self.phase_unwrapping_with_continuity(wrapped_pha, k1, k2)
        
        # 限制解包裹相位的范围
        max_phase = 32 * math.pi  # 5位格雷码的理论最大值
        unwrapped_pha = np.clip(unwrapped_pha, 0, max_phase)
        
        # 打印解包裹相位的范围
        print(f"连续性约束解包裹相位的范围: [{np.min(unwrapped_pha)}, {np.max(unwrapped_pha)}]")
        print(f"理论范围: [0, {32*math.pi}]")
        
        # 2. 评估相位质量
        print("\n步骤2: 评估相位质量...")
        quality = self.estimate_phase_quality(fringe_images)
        
        # 3. 可视化相位跳变区域
        print("\n步骤3: 可视化相位跳变区域...")
        jumps = self.visualize_phase_jumps(unwrapped_pha)
        
        # 4. 对解包裹相位进行平滑处理
        print("\n步骤4: 对解包裹相位进行平滑处理...")
        smoothed_pha = self.smooth_unwrapped_phase(unwrapped_pha)
        
        # 5. 可视化平滑后的相位跳变区域
        print("\n步骤5: 可视化平滑后的相位跳变区域...")
        smoothed_jumps = self.visualize_phase_jumps(smoothed_pha)
        
        # 比较平滑前后的跳变点数量
        jumps_before = np.sum(jumps > 0)
        jumps_after = np.sum(smoothed_jumps > 0)
        reduction_ratio = (jumps_before - jumps_after) / jumps_before * 100
        
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
        final_pha = self.optimize_phase_jumps(final_pha, smoothed_jumps, quality)
        
        # 打印最终解包裹相位的范围
        print(f"最终解包裹相位的范围: [{np.min(final_pha)}, {np.max(final_pha)}]")
        
        # 保存最终结果
        final_scaled = (final_pha * 255 / np.max(final_pha)).astype(np.uint8)
        final_color = cv.applyColorMap(final_scaled, cv.COLORMAP_JET)
        cv.imwrite("results/final_unwrapped_phase.png", final_color)
        
        return final_pha
    
    # 替换原始方法
    UnwrappedPhase.computeUnwrappedPhase = improved_computeUnwrappedPhase
    
    # 显示解包裹相位
    print("\n计算并显示解包裹相位...")
    unwrapped_phase = u.showUnwrappedPhase()
    
    # 分析解包裹相位的质量
    u.analyze_unwrapped_phase(unwrapped_phase)
