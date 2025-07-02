import numpy as np
import cv2 as cv
import math

class WrappedPhase():
    """
    包裹相位计算类
    
    该类实现了四步相移法计算包裹相位的核心算法。
    四步相移法通过投影四幅具有不同相移量的正弦条纹图案，
    然后根据采集到的四幅图像计算每个像素点的包裹相位值。
    
    包裹相位是指相位值被限制在[0, 2π]范围内的相位，
    需要通过解包裹算法才能获得连续的绝对相位。
    
    算法原理：
    对于四步相移法，投影的条纹图案强度可以表示为：
    I_j = A + B*cos(φ + j*π/2), j = 0,1,2,3
    其中A是背景光强，B是条纹对比度，φ是待求的相位值
    
    通过四幅图像可以计算出包裹相位：
    φ = arctan((I3 - I1)/(I0 - I2))
    """
    
    def __init__(self, n: int = 4):
        """
        初始化包裹相位计算器
        
        参数:
            n (int): 相移步数，默认为4（四步相移法）
                    四步相移法是最常用的相移方法，具有较好的抗噪声能力
        """
        self.n = n
    
    @staticmethod
    def getImageData(m: int = 4):
        """
        获取相机拍摄的n幅相移图像
        
        该方法从指定路径读取四步相移法采集的图像序列。
        图像文件名格式为"I5.png", "I6.png", "I7.png", "I8.png"，
        分别对应相移量为0, π/2, π, 3π/2的四幅图像。
        
        参数:
            m (int): 要读取的图像数量，默认为4
                    对应四步相移法的四幅图像
        
        返回:
            list: 包含m个numpy数组的列表，每个数组代表一幅图像
                  图像数据类型为uint8，尺寸为(height, width)
        
        注意:
            - 图像路径是硬编码的，需要根据实际情况修改
            - 图像应该按照相移顺序排列：0, π/2, π, 3π/2
            - 所有图像应该具有相同的尺寸和格式
        """
        I = []
        for i in range(m):
            # 构建图像文件路径，文件名从I5开始，对应相移序列
            filename = r"fringe_patterns\I" + str(i+5) + ".png"
            
            # 以二进制形式读取图像文件，避免中文路径问题
            img_file = np.fromfile(filename, dtype=np.uint8)
            
            # 使用imdecode解码图像，支持各种图像格式
            # flags=-1表示按原始格式读取（彩色图像为BGR，灰度图像为单通道）
            img = cv.imdecode(img_file, -1)
            
            I.append(img)
        return I

    def computeWrappedphase(self, I):
        """
        计算包裹相位
        
        该方法实现了四步相移法的核心算法，通过四幅相移图像计算包裹相位。
        算法考虑了四个象限的不同情况以及边界条件，确保相位计算的准确性。
        
        参数:
            I (list): 包含4个numpy数组的列表，每个数组代表一幅相移图像
                      图像顺序应该为：[I0, I1, I2, I3]，对应相移量[0, π/2, π, 3π/2]
        返回:
            numpy.ndarray: 包裹相位矩阵，数据类型为float32
                          相位值范围：[0, 2π]
                          尺寸：(height, width)
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
        
        # 将相位值缩放到[0,255]范围用于显示
        # 相位值范围[0,2π]映射到[0,255]
        pha_scaled = pha*255/(2*math.pi)
        pha_scaled1 = pha_scaled.astype(np.uint8)
        
        # 如果直接运行此文件，显示包裹相位图像
        if __name__ == "__main__":
            cv.imshow("Wrapped_Phase", pha_scaled1)
            key = cv.waitKey(0)
            if key == ord("s"):
                # 按's'键保存图像
                cv.imwrite("Wrapped_Phase.png", pha_scaled1)
            cv.destroyAllWindows()
        
        # 返回原始包裹相位矩阵（未缩放）
        return pha

if __name__ == "__main__":
    # 测试代码：创建包裹相位计算器实例并计算包裹相位
    w = WrappedPhase()
    w.computeWrappedphase(w.getImageData())