# 四步相移结构光三维重建项目结构分析

## 项目概述

本项目是一个基于"互补格雷码+四步相移码"方法的单目结构光三维重建系统。项目使用DLP投影仪投射结构光图案，通过灰度相机采集图像，然后通过算法处理获得被测物体的三维信息。

### 系统组成
- **DLP投影仪**: 闻亭PRO6500
- **灰度相机**: FLIR BFS-U3-50S5  
- **旋转平台**: 用于多角度采集

### 算法流程
1. 生成格雷码图像
2. 生成四步相移图像
3. 求解相对相位
4. 求解绝对相位
5. 获得相机-投影仪像素坐标之间的对应关系
6. 根据标定参数获得重建点云信息

## 项目文件结构

```
FourStepPhaseShifting-master/
├── src/
│   ├── python/                    # Python版本实现
│   │   ├── wrapped_phase_algorithm.py      # 包裹相位计算
│   │   ├── unwrapped_phase_algorithm.py    # 解包裹相位计算
│   │   ├── generate_graycode_map.py        # 格雷码生成与映射
│   │   ├── graycode_binarization.py        # 格雷码二值化处理
│   │   ├── generate_phase-shifting_map.py  # 相移图案生成
│   │   ├── gray_code.py                    # 简化版格雷码实现
│   │   ├── simplified_GrayCode_map.py      # 简化版格雷码映射
│   │   ├── phase_differences_map.py        # 相位差计算
│   │   └── 格雷光栅码值.txt                # 格雷码码值存储
│   └── matlab/                    # MATLAB版本实现
├── images/                        # 实验图片和结果
├── support/                       # 支持文档
└── README.md                      # 项目说明文档
```

## 核心功能模块分析

### 1. 包裹相位计算模块 (`wrapped_phase_algorithm.py`)

**主要类**: `WrappedPhase`

**核心功能**: 计算四步相移法得到的包裹相位

**主要函数**:
- `getImageData(m=4)`: 获取相机拍摄的n幅相移图
- `computeWrappedphase(I, width=1280, height=720)`: 计算包裹相位
  - 输入: 4幅相移图像
  - 输出: 包裹相位矩阵
  - 算法: 四步相移法，根据四个象限的不同情况计算相位

**算法特点**:
- 处理四个特殊位置（边界情况）
- 根据四个象限分别计算相位
- 输出相位范围: [0, 2π]

### 2. 解包裹相位计算模块 (`unwrapped_phase_algorithm.py`)

**主要类**: `UnwrappedPhase`

**核心功能**: 将包裹相位解包裹得到绝对相位

**主要函数**:
- `getBinarizedGrayCodes(m=5)`: 获得二值化后的格雷码图像
- `get_k1_k2()`: 获得k1和k2矩阵（格雷码解码）
- `computeUnwrappedPhase()`: 计算解包裹相位
- `showUnwrappedPhase()`: 显示解包裹相位

**算法特点**:
- 使用格雷码辅助解包裹
- 根据包裹相位的不同区间选择不同的解包裹策略
- 最终得到连续的绝对相位

### 3. 格雷码生成与映射模块 (`generate_graycode_map.py`)

**主要类**: `GrayCode`

**核心功能**: 生成格雷码图案和建立编码映射关系

**主要函数**:
- `__createGrayCode(n)`: 生成n位格雷码
- `__formCodes(n)`: 生成codes矩阵
- `toPattern(idx, cols=1920, rows=1080)`: 生成格雷码光栅图
- `__code2k(k)`: 将k映射到对应的格雷码
- `__k2v(k)`: 将k映射为v（十进制值）
- `store_gray_code_map_value()`: 存储格雷码映射值

**数据结构**:
- `codes`: 格雷码矩阵
- `code2k`: 格雷码到索引的映射字典
- `k2v`: 索引到十进制值的映射字典
- `v2k`: 十进制值到索引的映射字典

### 4. 格雷码二值化处理模块 (`graycode_binarization.py`)

**主要类**: `Binariization`

**核心功能**: 将格雷码图像进行二值化处理

**主要函数**:
- `get_threshold(m=4)`: 利用四幅相移图计算阈值
- `get_GC_images()`: 读取格雷码图片
- `getBinaryGrayCode()`: 将格雷码图像二值化处理

**算法特点**:
- 使用相移图像的平均值作为阈值
- 对格雷码图像进行二值化处理
- 输出二值化的格雷码图像

### 5. 相移图案生成模块 (`generate_phase-shifting_map.py`)

**主要类**: `PhaseShiftingCode`

**核心功能**: 生成四步相移图案

**主要函数**:
- `toPhasePattern(j, freq=16, width=1920, height=1080)`: 生成相移图案
  - 输入: 相移步数j，频率freq，图像尺寸
  - 输出: 相移图案
  - 公式: I = 128 + 127 * cos(2π * (i * freq / width + j/n))

### 6. 简化版格雷码模块

#### `gray_code.py`
- 简化版的格雷码实现
- 包含基本的格雷码生成和图案生成功能

#### `simplified_GrayCode_map.py`
- 在基础格雷码基础上增加了黑场和白场
- 用于系统标定和图像预处理

## 主要功能函数总结

### 相位计算相关
1. **`WrappedPhase.computeWrappedphase()`** - 四步相移包裹相位计算
2. **`UnwrappedPhase.computeUnwrappedPhase()`** - 格雷码辅助解包裹
3. **`UnwrappedPhase.get_k1_k2()`** - 格雷码解码得到k1、k2矩阵

### 图案生成相关
4. **`GrayCode.toPattern()`** - 格雷码图案生成
5. **`PhaseShiftingCode.toPhasePattern()`** - 相移图案生成
6. **`GrayCode.__createGrayCode()`** - 格雷码序列生成

### 图像处理相关
7. **`Binariization.getBinaryGrayCode()`** - 格雷码图像二值化
8. **`Binariization.get_threshold()`** - 阈值计算
9. **`WrappedPhase.getImageData()`** - 图像数据读取

### 数据映射相关
10. **`GrayCode.__code2k()`** - 格雷码到索引映射
11. **`GrayCode.__k2v()`** - 索引到十进制值映射
12. **`GrayCode.store_gray_code_map_value()`** - 映射关系存储

## 算法流程详解

### 1. 数据采集阶段
- 投影格雷码图案（5幅）
- 投影相移图案（4幅）
- 相机同步采集图像

### 2. 相位计算阶段
- 使用四步相移法计算包裹相位
- 处理边界情况和象限判断
- 输出范围在[0, 2π]的包裹相位

### 3. 格雷码解码阶段
- 对格雷码图像进行二值化处理
- 解码得到k1、k2矩阵
- 建立像素坐标到条纹序号的映射

### 4. 相位解包裹阶段
- 结合包裹相位和格雷码信息
- 根据相位区间选择解包裹策略
- 得到连续的绝对相位

### 5. 三维重建阶段
- 根据绝对相位计算深度信息
- 结合系统标定参数
- 生成三维点云数据

## 技术特点

1. **互补格雷码**: 提高解码精度和鲁棒性
2. **四步相移**: 标准的结构光相位计算方法
3. **模块化设计**: 各功能模块独立，便于维护和扩展
4. **Python实现**: 便于算法验证和快速原型开发

## 应用场景

- 工业三维测量
- 逆向工程
- 质量检测
- 机器人视觉
- 文化遗产数字化

## 参考文献

[1] Zhang Q, Su X, Xiang L, et al. 3-D shape measurement based on complementary Gray-code light[J]. Optics and Lasers in Engineering, 2012, 50(4): 574-579.

[2] 张启灿, 吴周杰. 基于格雷码图案投影的结构光三维成像技术[J]. 红外与激光工程, 2020, 49(3): 0303004-1-0303004-13. 