# 相位解包裹程序用户手册

## 1. 概述

本文档提供了相位解包裹程序的详细使用说明，包括独立版本(`unwrapped_phase_standalone.py`)和用户界面版本(`unwrapped_phase_ui.py`)。这两个程序用于处理四步相移图像和格雷码图像，计算解包裹相位，并可视化结果。

### 主要功能

- 四步相移法计算包裹相位
- 格雷码辅助的相位解包裹
- 相位质量评估和改进
- 相位跳变检测和修复
- 支持水平和垂直方向解包裹
- 多种可视化方式展示结果

## 2. 安装要求

### 2.1 依赖库

程序依赖以下Python库：

- numpy
- opencv-python (cv2)
- PySide6 (仅UI版本需要)

可以通过以下命令安装：

```bash
pip install numpy opencv-python
pip install PySide6  # 仅UI版本需要
```

### 2.2 文件结构

程序需要以下目录结构：

```
├── unwrapped_phase_standalone.py  # 独立版本
├── unwrapped_phase_ui.py          # UI版本
├── fringe_patterns/               # 存放四步相移图像
│   ├── I1.png                     # 水平方向相移图像
│   ├── I2.png
│   ├── I3.png
│   ├── I4.png
│   ├── I5.png                     # 垂直方向相移图像
│   ├── I6.png
│   ├── I7.png
│   └── I8.png
├── gray_patterns/                 # 存放格雷码图像
│   ├── gray_bit_0.png             # 原始格雷码图像
│   ├── gray_bit_1.png
│   ├── gray_bit_2.png
│   ├── gray_bit_3.png
│   ├── gray_bit_4.png
│   ├── matched_binary_0.png       # 二值化格雷码图像
│   ├── matched_binary_1.png
│   ├── matched_binary_2.png
│   ├── matched_binary_3.png
│   └── matched_binary_4.png
└── results/                       # 结果保存目录(自动创建)
```

## 3. 独立版本使用说明 (unwrapped_phase_standalone.py)

### 3.1 程序启动

直接运行Python脚本即可启动独立版本：

```bash
python unwrapped_phase_standalone.py
```

### 3.2 使用步骤

1. **选择显示选项**：
   - 选项1：仅显示最终结果
   - 选项2：显示关键中间过程（格雷码、四步相移图、K1/K2矩阵）
   - 选项3：显示所有处理过程（包括跳变检测、质量评估等）

2. **选择解包裹方向**：
   - 选项1：仅水平方向解包裹（使用I1-I4图像）
   - 选项2：仅垂直方向解包裹（使用I5-I8图像）
   - 选项3：水平和垂直方向解包裹并组合（默认）

3. **选择图像尺寸标准化方法**：
   - 选项1：自动裁剪到最小尺寸（默认）
   - 选项2：缩放所有图像到相同尺寸
   - 选项3：手动指定目标尺寸

4. **查看结果**：
   - 程序将显示解包裹相位图像和其他可视化结果
   - 鼠标点击图像可显示该点的相位值
   - 按's'键保存图像
   - 按'q'键或ESC键退出

5. **结果文件**：
   - 所有结果都保存在results目录中
   - 文件命名基于所选的解包裹方向和处理类型

### 3.3 测试数据生成

如果没有提供图像，程序会自动生成测试数据：

```python
u = UnwrappedPhase()
u.generate_test_patterns()
```

### 3.4 主要函数及功能

| 类/函数 | 描述 |
|---------|------|
| `normalize_image_size()` | 将一组图像标准化为相同尺寸，支持裁剪和缩放方法 |
| `GrayCode` 类 | 实现格雷码的生成、编码和解码功能 |
| `WrappedPhase` 类 | 实现四步相移法计算包裹相位的核心算法 |
| `Binariization` 类 | 处理格雷码图像的二值化 |
| `UnwrappedPhase` 类 | 实现相位解包裹的核心功能 |
| `computeUnwrappedPhase()` | 主要处理函数，计算解包裹相位 |
| `phase_unwrapping_with_continuity()` | 添加相位连续性约束的解包裹算法 |
| `estimate_phase_quality()` | 通过计算调制度评估相位质量 |
| `visualize_phase_jumps()` | 检测和可视化相位跳变区域 |
| `smooth_unwrapped_phase()` | 结合中值和高斯滤波进行相位平滑 |
| `optimize_phase_jumps()` | 优化相位跳变区域 |
| `combine_horizontal_vertical_phases()` | 组合水平和垂直方向的解包裹相位 |
| `get_k1_k2()` | 获取解包裹所需的k1和k2矩阵 |

## 4. UI版本使用说明 (unwrapped_phase_ui.py)

### 4.1 程序启动

直接运行Python脚本即可启动UI版本：

```bash
python unwrapped_phase_ui.py
```

### 4.2 界面说明

UI界面包含以下主要区域：

1. **控制面板**：
   - 文件夹选择区域
   - 解包裹方向选择
   - 显示模式选择
   - 图像尺寸标准化选项
   - 进度信息显示

2. **图像显示区**：
   - 相位图像显示
   - 鼠标交互功能

3. **信息面板**：
   - 相位数据统计
   - 文件信息
   - 操作按钮

### 4.3 使用步骤

1. **选择解包裹方向**：
   - 在"解包裹方向"下拉框中选择需要的方向：
     - 水平方向：仅使用I1-I4图像
     - 垂直方向：仅使用I5-I8图像
     - 水平+垂直组合：使用I1-I8图像，并组合结果

2. **选择图像文件夹**：
   - 点击"选择文件夹"按钮，选择包含四步相移图像的文件夹
   - 程序会检查文件夹中是否有足够的图像文件
   - 对于垂直方向或组合模式，需要至少8个图像文件

3. **设置图像尺寸标准化选项**：
   - 在"图像尺寸标准化"区域选择合适的方法：
     - 自动裁剪到最小尺寸
     - 缩放到相同尺寸
     - 手动指定目标尺寸

4. **开始处理**：
   - 点击"开始处理"按钮
   - 处理过程中，进度信息会显示在下方
   - 处理完成后，结果会自动显示在图像区域

5. **查看结果**：
   - 使用"显示模式"下拉框切换不同的显示方式：
     - 彩色相位图
     - 灰度相位图
     - 直方图均衡化相位图
     - 包裹相位图
     - K1矩阵图
     - K2矩阵图
   - 如果选择的是组合模式，还可以查看：
     - 水平方向解包裹相位图
     - 垂直方向解包裹相位图

6. **交互操作**：
   - 鼠标悬停在图像上可在状态栏显示相位值
   - 鼠标点击图像会在图像上标记该点并显示相位值
   - 在组合模式下，会同时显示水平和垂直方向的相位值

7. **保存结果**：
   - 所有结果自动保存在results目录中
   - 可以点击"保存当前视图"按钮保存当前显示的图像

### 4.4 主要类及功能

| 类/函数 | 描述 |
|---------|------|
| `UnwrappedPhaseApp` 类 | 主应用程序窗口类 |
| `ProcessingThread` 类 | 工作线程，避免UI冻结 |
| `CustomUnwrappedPhase` 类 | 继承自独立版本的UnwrappedPhase类，添加进度信号和UI交互 |
| `PhaseImageLabel` 类 | 自定义图像标签，支持鼠标交互和相位值显示 |
| `create_control_panel()` | 创建控制面板UI |
| `create_display_area()` | 创建图像显示区UI |
| `select_folder()` | 选择四步相移图像文件夹 |
| `start_processing()` | 开始处理流程 |
| `update_display()` | 更新图像显示 |
| `display_image()` | 将OpenCV图像显示在界面上 |
| `on_direction_changed()` | 当方向选择改变时更新显示选项 |
| `process_completed()` | 处理完成后的操作 |
| `get_fringe_images()` | 获取四步相移图像 |
| `compute_wrapped_phase()` | 计算包裹相位 |
| `compute_unwrapped_phase()` | 计算解包裹相位(改进版) |
| `save_unwrapped_phase_results()` | 保存解包裹相位结果 |
| `estimate_phase_quality()` | 估计相位质量 |
| `visualize_phase_jumps()` | 可视化相位跳变区域 |
| `smooth_unwrapped_phase()` | 对解包裹相位进行平滑处理 |
| `optimize_phase_jumps()` | 进一步优化相位跳变区域 |
| `combine_horizontal_vertical_phases()` | 组合水平和垂直方向的解包裹相位 |

## 5. 关键算法说明

### 5.1 四步相移法

四步相移法通过四幅具有固定相移量的图像计算包裹相位：

```
I₀(x,y) = A(x,y) + B(x,y)·cos[φ(x,y)]
I₁(x,y) = A(x,y) + B(x,y)·cos[φ(x,y) + π/2]
I₂(x,y) = A(x,y) + B(x,y)·cos[φ(x,y) + π]
I₃(x,y) = A(x,y) + B(x,y)·cos[φ(x,y) + 3π/2]
```

包裹相位计算公式：
```
φ(x,y) = arctan[(I₃(x,y) - I₁(x,y)) / (I₀(x,y) - I₂(x,y))]
```

### 5.2 格雷码辅助解包裹

格雷码是一种二进制编码，相邻数值之间只有一位二进制数不同。程序使用格雷码辅助确定相位周期数：

1. 生成n位格雷码图案
2. 对格雷码图像进行二值化处理
3. 解码二值化格雷码图像得到k1和k2矩阵
4. 使用k1和k2确定每个像素点所处的条纹周期
5. 结合包裹相位和周期信息计算解包裹相位

### 5.3 相位质量评估

通过计算调制度(modulation)评估相位质量：

```
modulation = sqrt((I₃-I₁)² + (I₀-I₂)²) / (I₀+I₁+I₂+I₃)
```

### 5.4 相位跳变检测与优化

1. 计算相位梯度，识别梯度异常大的区域为跳变点
2. 对跳变点，考虑其周围高质量区域的相位值
3. 使用质量加权平均计算新的相位值
4. 应用中值滤波进一步平滑跳变区域

### 5.5 水平和垂直相位组合

1. 分别计算水平和垂直方向的解包裹相位
2. 计算相位梯度，确定权重
3. 使用加权组合生成最终相位
4. 应用平滑处理改善结果

## 6. 常见问题解答

### 6.1 图像数量不足

- **问题**：程序提示"文件夹中只有X个图像文件，需要至少4个"
- **解决方法**：确保fringe_patterns文件夹中至少有4个相移图像(I1-I4)。对于垂直方向或组合模式，需要8个图像(I1-I8)。

### 6.2 图像尺寸不一致

- **问题**：程序提示"包裹相位和k1、k2的尺寸不一致"
- **解决方法**：使用图像尺寸标准化功能，选择"自动裁剪到最小尺寸"或"缩放到相同尺寸"。

### 6.3 相位跳变严重

- **问题**：解包裹相位中有明显的跳变和不连续
- **解决方法**：
  - 增大平滑核尺寸(默认为5)
  - 调整相位跳变检测阈值(默认为0.5)
  - 确保格雷码图像清晰且正确曝光

### 6.4 无法选择解包裹方向

- **问题**：UI版本中方向选择下拉框被禁用
- **解决方法**：确保界面初始化正确，使用最新版本的程序。现在的版本支持在选择文件夹前直接选择解包裹方向。

### 6.5 相位图像模糊或不清晰

- **问题**：生成的相位图像模糊或质量较低
- **解决方法**：
  - 确保输入图像质量高，光照均匀
  - 尝试使用不同的尺寸标准化方法
  - 调整平滑参数，减少过度平滑
  - 使用直方图均衡化模式查看以增强对比度

## 7. 高级用法

### 7.1 自定义图像处理参数

独立版本可以通过修改源代码自定义参数：

```python
# 示例：修改平滑核大小
smoothed_pha = self.smooth_unwrapped_phase(unwrapped_pha, kernel_size=7)  # 默认为5

# 示例：修改跳变检测阈值
jumps = self.visualize_phase_jumps(unwrapped_pha, threshold=0.3)  # 默认为0.5
```

### 7.2 集成到其他项目

可以将相位解包裹算法集成到其他项目中：

```python
from unwrapped_phase_standalone import UnwrappedPhase

# 创建解包裹相位计算器实例
unwrapper = UnwrappedPhase()

# 自定义图像处理参数
unwrapper.standard_size = (480, 640)  # 设置标准尺寸
unwrapper.size_method = "resize"      # 设置尺寸调整方法

# 计算解包裹相位
unwrapped_phase = unwrapper.computeUnwrappedPhase(
    show_details=False,
    direction="horizontal"
)

# 使用解包裹相位进行进一步处理
# ...
```

### 7.3 自定义可视化

可以根据需要修改可视化方式：

```python
# 将相位值缩放到[0,255]范围
scaled = (unwrapped_phase * 255 / np.max(unwrapped_phase)).astype(np.uint8)

# 应用不同的颜色映射
# COLORMAP_JET - 默认，彩虹色
# COLORMAP_VIRIDIS - 改进的彩虹色，更好的感知均匀性
# COLORMAP_INFERNO - 黑到黄的热力图
# COLORMAP_PLASMA - 深紫到黄
color_map = cv.applyColorMap(scaled, cv.COLORMAP_VIRIDIS)
```

## 8. 附录

### 8.1 相位解包裹的数学原理

包裹相位的范围被限制在[0, 2π)，而实际相位可能跨越多个周期：

```
Φ(x,y) = φ(x,y) + 2πk(x,y)
```

其中:
- Φ(x,y)是解包裹相位（绝对相位）
- φ(x,y)是包裹相位（范围在[0, 2π)）
- k(x,y)是整数周期数

格雷码辅助方法通过额外的二进制编码图案确定k(x,y)的值。

### 8.2 输入图像要求

- **四步相移图像**：相移量分别为0, π/2, π, 3π/2
- **格雷码图像**：通常使用5位格雷码，共需要5幅图像
- **图像格式**：支持PNG, JPG, BMP, TIF等常见格式
- **图像尺寸**：推荐分辨率不低于640x480
- **文件命名**：水平方向I1-I4，垂直方向I5-I8

### 8.3 结果文件说明

| 文件名 | 描述 |
|--------|------|
| `*_unwrapped_phase_original.png` | 原始缩放的解包裹相位图 |
| `*_unwrapped_phase_color.png` | 应用伪彩色映射的解包裹相位图 |
| `*_unwrapped_phase_equalized.png` | 直方图均衡化后的解包裹相位图 |
| `*_unwrapped_phase_height.png` | 用于3D可视化的高度图 |
| `phase_quality.png` | 相位质量图(调制度) |
| `phase_jumps_binary.png` | 二值化的相位跳变图 |
| `phase_jumps_color.png` | 彩色的相位跳变图 |
| `phase_with_jumps.png` | 叠加跳变区域的相位图 |
| `optimization_difference.png` | 优化前后的差异图 |
| `optimized_phase.png` | 优化后的相位图 |
| `phase_distribution_map.png` | 二维相位分布图(组合模式) |
| `phase_contour_map.png` | 相位等值线图(组合模式) |
| `phase_3d_distribution.png` | 3D相位分布图(组合模式) |

其中，文件名前缀为`horizontal_`、`vertical_`或`combined_`，分别表示水平方向、垂直方向或组合模式的结果。 