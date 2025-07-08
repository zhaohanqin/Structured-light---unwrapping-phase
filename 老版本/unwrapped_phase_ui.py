import sys
import os
import numpy as np
import cv2 as cv
import math
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QFileDialog, QComboBox, QGroupBox, QSplitter,
                             QCheckBox, QScrollArea, QFrame, QMessageBox, QGridLayout, QStatusBar,
                             QProgressBar, QLineEdit, QToolButton)
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QIcon, QFontDatabase, QIntValidator
from PySide6.QtCore import Qt, QThread, Signal, QSize, QPropertyAnimation, QEasingCurve, Property

# 导入解相位算法
from unwrapped_phase import UnwrappedPhase, WrappedPhase, GrayCode

# 设置应用全局样式
def set_app_style(app):
    """设置应用程序的全局样式"""
    # 应用程序基本样式
    app.setStyle("Fusion")
    
    # 加载样式表
    style = """
    /* 全局样式 */
    QMainWindow, QDialog {
        background-color: #F9FBFC;
    }
    
    QWidget {
        font-family: 'Segoe UI', 'Microsoft YaHei';
        font-size: 11pt;
        color: #34495E;
    }
    
    /* 按钮样式 */
    QPushButton {
        background-color: #3498DB;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: bold;
    }
    
    QPushButton:hover {
        background-color: #2980B9;
    }
    
    QPushButton:pressed {
        background-color: #1F6AA5;
    }
    
    QPushButton:disabled {
        background-color: #BDC3C7;
        color: #7F8C8D;
    }
    
    /* 主要操作按钮 */
    QPushButton#primaryButton {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498DB, stop:1 #2980B9);
        padding: 10px 20px;
        font-size: 12pt;
    }
    
    QPushButton#primaryButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2980B9, stop:1 #1F6AA5);
    }
    
    /* 次要按钮 */
    QPushButton#secondaryButton {
        background-color: #FFFFFF;
        color: #3498DB;
        border: 1px solid #3498DB;
    }
    
    QPushButton#secondaryButton:hover {
        background-color: #EBF5FB;
    }
    
    /* 组合框样式 */
    QComboBox {
        background-color: white;
        border: 1px solid #BDC3C7;
        border-radius: 6px;
        padding: 6px 12px;
    }
    
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: right center;
        width: 20px;
        border-left: none;
    }
    
    QComboBox QAbstractItemView {
        background-color: white;
        border: 1px solid #BDC3C7;
        border-radius: 0px;
        selection-background-color: #3498DB;
        selection-color: white;
    }
    
    /* 分组框样式 */
    QGroupBox {
        font-weight: bold;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        margin-top: 16px;
        background-color: white;
        padding: 12px;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 8px;
        color: #34495E;
    }
    
    /* 标签样式 */
    QLabel {
        color: #34495E;
    }
    
    QLabel#infoLabel {
        background-color: #F5F7FA;
        border-radius: 6px;
        padding: 8px;
    }
    
    QLabel#headingLabel {
        font-size: 13pt;
        font-weight: bold;
        color: #2C3E50;
    }
    
    /* 复选框样式 */
    QCheckBox {
        spacing: 8px;
    }
    
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border: 1px solid #BDC3C7;
        border-radius: 4px;
    }
    
    QCheckBox::indicator:checked {
        background-color: #3498DB;
        border-color: #3498DB;
    }
    
    /* 状态栏样式 */
    QStatusBar {
        background-color: #F5F7FA;
        color: #34495E;
        border-top: 1px solid #E0E0E0;
    }
    
    /* 滚动区域样式 */
    QScrollArea {
        background-color: transparent;
        border: none;
    }
    
    QScrollBar:vertical {
        border: none;
        background-color: #F5F7FA;
        width: 12px;
        margin: 0px;
    }
    
    QScrollBar::handle:vertical {
        background-color: #BDC3C7;
        border-radius: 6px;
        min-height: 30px;
    }
    
    QScrollBar::handle:vertical:hover {
        background-color: #3498DB;
    }
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    
    /* 进度条样式 */
    QProgressBar {
        border: none;
        background-color: #F5F7FA;
        border-radius: 6px;
        text-align: center;
        color: #34495E;
    }
    
    QProgressBar::chunk {
        background-color: #3498DB;
        border-radius: 6px;
    }
    
    /* 分割器样式 */
    QSplitter::handle {
        background-color: #E0E0E0;
    }
    
    QSplitter::handle:horizontal {
        width: 2px;
    }
    
    QSplitter::handle:vertical {
        height: 2px;
    }
    
    /* 框架样式 */
    QFrame#infoPanel {
        background-color: white;
        border-radius: 8px;
        border: 1px solid #E0E0E0;
    }
    """
    
    app.setStyleSheet(style)
    return app

# 工作线程，避免UI冻结
class ProcessingThread(QThread):
    # 定义信号
    progress_signal = Signal(str)  # 进度信息
    result_signal = Signal(np.ndarray)  # 解相位结果
    wrapped_signal = Signal(np.ndarray)  # 包裹相位结果
    k1_signal = Signal(np.ndarray)  # k1矩阵
    k2_signal = Signal(np.ndarray)  # k2矩阵
    fringe_signal = Signal(list)  # 四步相移图像
    horizontal_phase_signal = Signal(np.ndarray)  # 水平方向解包裹相位
    vertical_phase_signal = Signal(np.ndarray)  # 垂直方向解包裹相位
    finished_signal = Signal()  # 完成信号
    error_signal = Signal(str)  # 错误信号
    
    def __init__(self, fringe_folder, save_intermediate=True, normalization_params=None, direction="horizontal"):
        super().__init__()
        self.fringe_folder = fringe_folder
        self.save_intermediate = save_intermediate
        self.normalization_params = normalization_params or {"method": "crop", "target_size": None}
        self.direction = direction  # 解包裹方向：horizontal, vertical 或 combined
        
    def run(self):
        try:
            self.progress_signal.emit("开始处理...")
            
            # 设置输入输出文件夹
            os.environ["FRINGE_FOLDER"] = self.fringe_folder
            
            # 确保结果文件夹存在
            if not os.path.exists("results"):
                os.makedirs("results")
                self.progress_signal.emit("创建了results目录")
                
            if not os.path.exists("gray_patterns"):
                os.makedirs("gray_patterns")
                self.progress_signal.emit("创建了gray_patterns目录")
            
            # 检查图像数量是否足够
            image_files = [f for f in os.listdir(self.fringe_folder) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
            
            # 根据选择的方向和图像数量决定处理方式
            if len(image_files) < 4:
                self.error_signal.emit(f"文件夹中只有{len(image_files)}个图像文件，至少需要4个才能进行解包裹")
                return
                
            if (self.direction == "vertical" or self.direction == "combined") and len(image_files) < 8:
                self.progress_signal.emit(f"警告: 您选择了{self.direction}方向，但只有{len(image_files)}个图像，将强制切换到水平方向")
                self.direction = "horizontal"
            
            # 创建解包裹相位计算器实例
            self.progress_signal.emit("初始化解包裹相位计算器...")
            unwrapper = CustomUnwrappedPhase(self.progress_signal, normalization_params=self.normalization_params)
            
            # 根据选择的方向进行解包裹
            if self.direction == "combined":
                self.progress_signal.emit("选择了水平+垂直组合解包裹模式...")
                
                # 1. 水平方向解包裹
                self.progress_signal.emit("执行水平方向解包裹...")
                horizontal_phase = self.process_single_direction(unwrapper, "horizontal")
                
                # 发送水平方向相位信号
                if horizontal_phase is not None:
                    self.horizontal_phase_signal.emit(horizontal_phase)
                
                # 2. 垂直方向解包裹
                self.progress_signal.emit("执行垂直方向解包裹...")
                vertical_phase = self.process_single_direction(unwrapper, "vertical")
                
                # 发送垂直方向相位信号
                if vertical_phase is not None:
                    self.vertical_phase_signal.emit(vertical_phase)
                
                # 3. 组合两个方向的结果
                self.progress_signal.emit("组合水平和垂直方向的解包裹结果...")
                if horizontal_phase is not None and vertical_phase is not None:
                    # 调用combine_horizontal_vertical_phases方法
                    combined_phase = unwrapper.combine_horizontal_vertical_phases(horizontal_phase, vertical_phase)
                    
                    # 发送组合结果
                    self.result_signal.emit(combined_phase)
                    self.progress_signal.emit("处理完成，已生成组合解包裹相位")
                else:
                    self.error_signal.emit("无法组合水平和垂直方向的解包裹结果，请确保两个方向都有足够的图像")
            else:
                # 单一方向的解包裹
                phase = self.process_single_direction(unwrapper, self.direction)
                if phase is not None:
                    # 发送结果
                    self.result_signal.emit(phase)
                    
                    # 同时更新对应方向的相位
                    if self.direction == "horizontal":
                        self.horizontal_phase_signal.emit(phase)
                    elif self.direction == "vertical":
                        self.vertical_phase_signal.emit(phase)
                    
                    self.progress_signal.emit(f"处理完成，已生成{self.direction}方向的解包裹相位")
                else:
                    self.error_signal.emit(f"处理{self.direction}方向的解包裹相位失败")
            
            self.finished_signal.emit()
            
        except Exception as e:
            import traceback
            self.error_signal.emit(f"处理过程中发生错误: {str(e)}\n{traceback.format_exc()}")
            
    def process_single_direction(self, unwrapper, direction):
        """处理单一方向的解包裹"""
        try:
            # 获取四步相移图像
            self.progress_signal.emit(f"读取{direction}方向的四步相移图像...")
            fringe_images = unwrapper.get_fringe_images(direction=direction)
            self.fringe_signal.emit(fringe_images)
            
            # 计算包裹相位
            self.progress_signal.emit(f"计算{direction}方向的包裹相位...")
            wrapped_phase = unwrapper.compute_wrapped_phase(fringe_images)
            self.wrapped_signal.emit(wrapped_phase)
            
            # 获取k1和k2矩阵
            self.progress_signal.emit(f"解码{direction}方向的格雷码...")
            k1, k2 = unwrapper.get_k1_k2()
            self.k1_signal.emit(k1)
            self.k2_signal.emit(k2)
            
            # 计算解包裹相位
            self.progress_signal.emit(f"计算{direction}方向的解包裹相位...")
            unwrapped_phase = unwrapper.compute_unwrapped_phase(wrapped_phase, k1, k2, self.save_intermediate)
            
            return unwrapped_phase
        except Exception as e:
            self.progress_signal.emit(f"处理{direction}方向时出错: {str(e)}")
            return None


# 自定义的解包裹相位类，添加进度信号
class CustomUnwrappedPhase(UnwrappedPhase):
    def __init__(self, progress_signal=None, normalization_params=None):
        method = normalization_params["method"] if normalization_params else "crop"
        target_size = normalization_params["target_size"] if normalization_params else None
        super().__init__()
        self.progress_signal = progress_signal
        # 传递标准化参数到父类
        self.size_method = method
        self.standard_size = target_size
        self.direction = "horizontal"  # 默认方向为水平
        
    def emit_progress(self, message):
        if self.progress_signal:
            # 高亮尺寸警告
            if "尺寸不一致" in message or "警告" in message:
                message = f"<span style='color:#e67e22;font-weight:bold'>{message}</span>"
            self.progress_signal.emit(message)
            
    def get_fringe_images(self, direction="horizontal"):
        """获取四步相移图像"""
        self.emit_progress("读取四步相移图像...")
        self.direction = direction
        
        fringe_folder = os.environ.get("FRINGE_FOLDER", "fringe_patterns")
        I = []
        
        # 查找文件夹中的所有图像文件
        image_files = [f for f in os.listdir(fringe_folder) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        
        # 确保至少有4个图像文件
        if len(image_files) < 4:
            self.emit_progress(f"警告: 在{fringe_folder}中只找到{len(image_files)}个图像文件，需要至少4个")
            raise ValueError(f"文件夹{fringe_folder}中的图像文件数量不足，需要至少4个")
        
        # 根据方向选择图像文件
        if direction == "horizontal":
            # 选择前4个图像文件用于水平方向相位
            selected_files = sorted(image_files)[:4]
            self.emit_progress(f"使用水平方向图像文件: {', '.join(selected_files)}")
        elif direction == "vertical":
            # 选择5-8个图像文件用于垂直方向相位
            if len(image_files) < 8:
                self.emit_progress(f"警告: 在{fringe_folder}中只找到{len(image_files)}个图像文件，垂直方向需要至少8个")
                self.emit_progress(f"尝试使用现有图像文件代替...")
                
                if len(image_files) >= 4:
                    # 如果至少有4个图像，使用最后4个
                    selected_files = sorted(image_files)[-4:]
                    self.emit_progress(f"使用最后4个图像作为垂直方向: {', '.join(selected_files)}")
                else:
                    raise ValueError(f"文件夹{fringe_folder}中没有足够的图像用于垂直方向解包裹，需要至少4个")
            else:
                selected_files = sorted(image_files)[4:8]
                self.emit_progress(f"使用垂直方向图像文件: {', '.join(selected_files)}")
        else:
            # 默认使用前4个文件
            selected_files = sorted(image_files)[:4]
            self.emit_progress(f"使用图像文件: {', '.join(selected_files)}")
        
        for filename in selected_files:
            filepath = os.path.join(fringe_folder, filename)
            try:
                # 读取图像
                img = cv.imread(filepath, 0)  # 0表示以灰度模式读取
                if img is None:
                    raise ValueError(f"无法读取图像: {filepath}")
                I.append(img)
                self.emit_progress(f"成功读取图像: {filename}")
            except Exception as e:
                self.emit_progress(f"读取图像{filename}时出错: {str(e)}")
                raise
                
        # 标准化所有图像的尺寸
        from unwrapped_phase import normalize_image_size
        I = normalize_image_size(I, self.standard_size, self.size_method)
        if len(I) > 0 and I[0] is not None:
            self.emit_progress(f"相移图像尺寸: {I[0].shape[0]}x{I[0].shape[1]} ({direction}方向)")
            
        return I
        
    def compute_wrapped_phase(self, I):
        """计算包裹相位"""
        self.emit_progress("计算包裹相位...")
        
        # 创建包裹相位计算器实例
        wp = WrappedPhase()
        
        # 使用传入的图像计算包裹相位
        wrapped_pha = wp.computeWrappedphase(I)
        
        # 打印包裹相位的范围
        min_phase = np.min(wrapped_pha)
        max_phase = np.max(wrapped_pha)
        self.emit_progress(f"包裹相位范围: [{min_phase}, {max_phase}]")
        
        return wrapped_pha
        
    def compute_unwrapped_phase(self, wrapped_pha, k1, k2, save_intermediate=True, quality_analysis=True):
        """计算解包裹相位（改进版）"""
        self.emit_progress("开始解包裹相位计算...")
        
        # 检查尺寸是否一致
        if wrapped_pha.shape != k1.shape or wrapped_pha.shape != k2.shape:
            self.emit_progress("警告: 包裹相位和k1、k2的尺寸不一致，尝试调整...")
            min_rows = min(wrapped_pha.shape[0], k1.shape[0], k2.shape[0])
            min_cols = min(wrapped_pha.shape[1], k1.shape[1], k2.shape[1])
            
            wrapped_pha = wrapped_pha[:min_rows, :min_cols]
            k1 = k1[:min_rows, :min_cols]
            k2 = k2[:min_rows, :min_cols]
            
            self.emit_progress(f"调整后尺寸: {min_rows}x{min_cols}")
        
        # 1. 使用相位连续性约束的解包裹算法
        self.emit_progress("步骤1: 使用相位连续性约束的解包裹算法...")
        unwrapped_pha = self.phase_unwrapping_with_continuity(wrapped_pha, k1, k2)
        
        # 限制解包裹相位的范围
        max_phase = 32 * math.pi  # 5位格雷码的理论最大值
        unwrapped_pha = np.clip(unwrapped_pha, 0, max_phase)
        
        min_unwrapped = np.min(unwrapped_pha)
        max_unwrapped = np.max(unwrapped_pha)
        self.emit_progress(f"连续性约束解包裹相位的范围: [{min_unwrapped}, {max_unwrapped}]")
        
        # 如果不需要质量分析和改进，直接返回基本结果
        if not quality_analysis:
            if not save_intermediate:
                return unwrapped_pha
                
            self.save_unwrapped_phase_results(unwrapped_pha)
            return unwrapped_pha
            
        try:
            # 获取相移图像用于质量分析（如果已经有）
            fringe_images = getattr(self, '_fringe_images', None)
            
            # 如果没有相移图像，尝试重新读取
            if fringe_images is None:
                self.emit_progress("重新读取相移图像用于质量分析...")
                fringe_images = self.get_fringe_images(direction=self.direction)
                self._fringe_images = fringe_images
                
            # 2. 评估相位质量
            self.emit_progress("步骤2: 评估相位质量...")
            quality = self.estimate_phase_quality(fringe_images, show_details=False)
            
            # 3. 可视化相位跳变区域
            self.emit_progress("步骤3: 检测相位跳变区域...")
            jumps = self.visualize_phase_jumps(unwrapped_pha, threshold=0.5, show_details=False)
            
            # 4. 对解包裹相位进行平滑处理
            self.emit_progress("步骤4: 对解包裹相位进行平滑处理...")
            smoothed_pha = self.smooth_unwrapped_phase(unwrapped_pha)
            
            # 5. 可视化平滑后的相位跳变区域
            self.emit_progress("步骤5: 检测平滑后的相位跳变区域...")
            smoothed_jumps = self.visualize_phase_jumps(smoothed_pha, threshold=0.5, show_details=False)
            
            # 比较平滑前后的跳变点数量
            jumps_before = np.sum(jumps > 0)
            jumps_after = np.sum(smoothed_jumps > 0)
            if jumps_before > 0:
                reduction_ratio = (jumps_before - jumps_after) / jumps_before * 100
            else:
                reduction_ratio = 0
                
            self.emit_progress(f"平滑处理效果: 平滑前跳变点{jumps_before}个，平滑后{jumps_after}个，减少比例{reduction_ratio:.2f}%")
            
            # 6. 综合质量评估，生成最终解包裹相位
            self.emit_progress("步骤6: 综合质量评估，优化相位...")
            
            # 使用质量图作为权重，对低质量区域进行特殊处理
            low_quality_mask = quality < 0.2
            
            # 对低质量区域使用更大的平滑核
            final_pha = smoothed_pha.copy()
            if np.any(low_quality_mask):
                # 对低质量区域使用更大的平滑核
                large_kernel_size = 9
                extra_smoothed = cv.GaussianBlur(smoothed_pha, (large_kernel_size, large_kernel_size), 0)
                final_pha[low_quality_mask] = extra_smoothed[low_quality_mask]
                
                self.emit_progress(f"对{np.sum(low_quality_mask)}个低质量像素点进行了额外平滑处理")
            
            # 7. 优化相位跳变区域
            self.emit_progress("步骤7: 优化相位跳变区域...")
            final_pha = self.optimize_phase_jumps(final_pha, smoothed_jumps, quality, show_details=False)
            
            # 打印最终解包裹相位的范围
            min_final = np.min(final_pha)
            max_final = np.max(final_pha)
            self.emit_progress(f"最终优化后的解包裹相位范围: [{min_final}, {max_final}]")
            
            # 如果不需要保存中间结果，直接返回
            if not save_intermediate:
                return final_pha
            
            # 保存结果
            self.save_unwrapped_phase_results(final_pha)
            return final_pha
            
        except Exception as e:
            import traceback
            self.emit_progress(f"<span style='color:red'>质量分析和优化过程中出错: {str(e)}</span>")
            self.emit_progress(traceback.format_exc())
            self.emit_progress("使用基本解包裹结果继续...")
            
            # 如果不需要保存中间结果，直接返回
            if not save_intermediate:
                return unwrapped_pha
                
            # 保存基本结果
            self.save_unwrapped_phase_results(unwrapped_pha)
            return unwrapped_pha
            
    def save_unwrapped_phase_results(self, unwrapped_pha):
        """保存解包裹相位结果"""
        self.emit_progress("保存结果...")
        
        # 创建results目录（如果不存在）
        if not os.path.exists("results"):
            os.makedirs("results")
        
        # 将相位值缩放到[0,255]范围用于显示和保存
        # 对于5位格雷码，相位范围约为[0, 32π]
        upha_scaled = np.rint(unwrapped_pha*255/(32*math.pi))
        upha_scaled_uint = upha_scaled.astype(np.uint8)
        
        # 使用英文代替中文方向表示，避免编码问题
        if hasattr(self, 'direction'):
            if self.direction == "horizontal":
                prefix = "horizontal_"
            elif self.direction == "vertical":
                prefix = "vertical_"
            elif self.direction == "combined":
                prefix = "combined_"
            else:
                prefix = ""
        else:
            prefix = ""
        
        # 1. 原始缩放视图
        cv.imwrite(f"results/{prefix}unwrapped_phase_original.png", upha_scaled_uint)
        
        # 2. 应用伪彩色映射以增强可视化效果
        upha_color = cv.applyColorMap(upha_scaled_uint, cv.COLORMAP_JET)
        cv.imwrite(f"results/{prefix}unwrapped_phase_color.png", upha_color)
        
        # 3. 应用直方图均衡化以增强对比度
        upha_eq = cv.equalizeHist(upha_scaled_uint)
        cv.imwrite(f"results/{prefix}unwrapped_phase_equalized.png", upha_eq)
        
        # 4. 3D可视化（保存为高度图）
        # 将相位值归一化到[0,1]范围
        upha_norm = unwrapped_pha / np.max(unwrapped_pha)
        # 保存为16位PNG，以保留更多细节
        cv.imwrite(f"results/{prefix}unwrapped_phase_height.png", (upha_norm * 65535).astype(np.uint16))
        
        self.emit_progress(f"解包裹相位计算完成，结果已保存到results目录")
            
    def estimate_phase_quality(self, fringe_images, show_details=False):
        """
        估计相位质量
        
        该方法通过计算调制度来评估每个像素点的相位质量。
        调制度是衡量相位质量的重要指标，它反映了条纹对比度和信噪比。
        调制度越高，相位质量越好。
        
        调制度计算公式: sqrt((I3-I1)^2 + (I0-I2)^2) / (I0+I1+I2+I3)
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
        
        # 使用英文代替中文方向表示，避免编码问题
        if hasattr(self, 'direction'):
            if self.direction == "horizontal":
                prefix = "horizontal_"
            elif self.direction == "vertical":
                prefix = "vertical_"
            elif self.direction == "combined":
                prefix = "combined_"
            else:
                prefix = ""
        else:
            prefix = ""
        
        cv.imwrite(f"results/{prefix}phase_quality.png", quality_color)
        
        # 计算质量统计信息
        quality_mean = np.mean(modulation)
        quality_std = np.std(modulation)
        quality_min = np.min(modulation)
        quality_max = np.max(modulation)
        
        # 计算低质量区域的比例
        low_quality_threshold = 0.2
        low_quality_ratio = np.sum(modulation < low_quality_threshold) / modulation.size
        
        self.emit_progress(f"相位质量分析: 平均质量={quality_mean:.4f}, 最小质量={quality_min:.4f}, 最大质量={quality_max:.4f}")
        self.emit_progress(f"低质量区域比例(<{low_quality_threshold}): {low_quality_ratio*100:.2f}%")
        
        return modulation
        
    def visualize_phase_jumps(self, unwrapped_pha, threshold=0.5, show_details=False):
        """
        可视化相位跳变区域
        
        该方法检测相位梯度过大的区域，识别可能的相位跳变点。
        相位跳变是解包裹相位中的常见问题，通常表现为相邻像素之间
        的相位值差异异常大。
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
        
        self.emit_progress(f"相位跳变分析: 检测到{jump_count}个跳变点，占总像素比例{jump_ratio*100:.4f}%")
        
        # 保存二值化跳变图像
        if not os.path.exists("results"):
            os.makedirs("results")
        
        # 使用英文代替中文方向表示，避免编码问题
        if hasattr(self, 'direction'):
            if self.direction == "horizontal":
                prefix = "horizontal_"
            elif self.direction == "vertical":
                prefix = "vertical_"
            elif self.direction == "combined":
                prefix = "combined_"
            else:
                prefix = ""
        else:
            prefix = ""
        
        cv.imwrite(f"results/{prefix}phase_jumps_binary.png", jumps)
        
        # 创建彩色跳变图像，红色表示跳变区域
        jumps_color = np.zeros((jumps.shape[0], jumps.shape[1], 3), dtype=np.uint8)
        jumps_color[jumps > 0] = [0, 0, 255]  # 红色表示跳变区域
        cv.imwrite(f"results/{prefix}phase_jumps_color.png", jumps_color)
        
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
        cv.imwrite(f"results/{prefix}phase_with_jumps.png", phase_with_jumps)
        
        return jumps
        
    def smooth_unwrapped_phase(self, unwrapped_pha, kernel_size=5):
        """
        对解包裹相位进行平滑处理
        
        该方法结合中值滤波和高斯滤波，对解包裹相位进行平滑处理，
        以减少噪声和相位跳变。中值滤波主要用于去除离群值（如相位跳变点），
        而高斯滤波则用于保留相位的整体结构，同时减少小的波动。
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
        
    def optimize_phase_jumps(self, unwrapped_pha, jumps, quality, show_details=False):
        """
        进一步优化相位跳变区域
        
        该方法针对检测到的相位跳变区域进行特殊处理，
        通过考虑周围高质量区域的相位值，修复跳变区域的相位值，
        提高相位的连续性和准确性。
        """
        # 创建优化后的相位矩阵
        optimized_pha = unwrapped_pha.copy()
        
        # 获取跳变区域的坐标
        jump_coords = np.where(jumps > 0)
        
        if len(jump_coords[0]) == 0:
            self.emit_progress("没有检测到相位跳变区域，无需优化")
            return optimized_pha
            
        self.emit_progress(f"检测到{len(jump_coords[0])}个相位跳变点，正在优化...")
        
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
        # 使用英文代替中文方向表示，避免编码问题
        if hasattr(self, 'direction'):
            if self.direction == "horizontal":
                prefix = "horizontal_"
            elif self.direction == "vertical":
                prefix = "vertical_"
            elif self.direction == "combined":
                prefix = "combined_"
            else:
                prefix = ""
        else:
            prefix = ""
        
        cv.imwrite(f"results/{prefix}optimization_difference.png", diff_color)
        
        # 可视化优化后的相位
        optimized_scaled = (optimized_pha * 255 / np.max(optimized_pha)).astype(np.uint8)
        optimized_color = cv.applyColorMap(optimized_scaled, cv.COLORMAP_JET)
        cv.imwrite(f"results/{prefix}optimized_phase.png", optimized_color)
        
        # 检查优化后的相位跳变
        optimized_jumps = self.visualize_phase_jumps(optimized_pha, threshold=0.5, show_details=False)
        
        # 比较优化前后的跳变点数量
        jumps_before = np.sum(jumps > 0)
        jumps_after = np.sum(optimized_jumps > 0)
        if jumps_before > 0:
            reduction_ratio = (jumps_before - jumps_after) / jumps_before * 100
        else:
            reduction_ratio = 0
        
        self.emit_progress(f"优化处理效果: 优化前跳变点{jumps_before}个，优化后{jumps_after}个，减少比例{reduction_ratio:.2f}%")
        
        if np.max(diff) > 0:
            self.emit_progress(f"最大相位差异: {np.max(diff):.6f} rad，平均相位差异: {np.mean(diff):.6f} rad")
        
        return optimized_pha
        
    def combine_horizontal_vertical_phases(self, horizontal_phase, vertical_phase):
        """
        组合水平和垂直方向的解包裹相位
        
        该方法将水平和垂直方向的相位信息进行组合，利用两者的优势，
        提高解包裹相位的质量和精度。水平方向通常在垂直边缘处表现更好，
        而垂直方向则在水平边缘处表现更好。
        
        参数:
            horizontal_phase: 水平方向的解包裹相位
            vertical_phase: 垂直方向的解包裹相位
            
        返回:
            组合后的解包裹相位
        """
        self.emit_progress("开始组合水平和垂直方向的解包裹相位...")
        
        # 确保两个相位矩阵尺寸一致
        if horizontal_phase.shape != vertical_phase.shape:
            self.emit_progress("警告: 水平和垂直相位尺寸不一致，尝试调整...")
            min_rows = min(horizontal_phase.shape[0], vertical_phase.shape[0])
            min_cols = min(horizontal_phase.shape[1], vertical_phase.shape[1])
            
            horizontal_phase = horizontal_phase[:min_rows, :min_cols]
            vertical_phase = vertical_phase[:min_rows, :min_cols]
            self.emit_progress(f"调整后尺寸: {min_rows}x{min_cols}")
        
        # 计算水平和垂直方向的相位梯度
        h_grad_x = np.abs(np.diff(horizontal_phase, axis=1, append=horizontal_phase[:, :1]))
        h_grad_y = np.abs(np.diff(horizontal_phase, axis=0, append=horizontal_phase[:1, :]))
        v_grad_x = np.abs(np.diff(vertical_phase, axis=1, append=vertical_phase[:, :1]))
        v_grad_y = np.abs(np.diff(vertical_phase, axis=0, append=vertical_phase[:1, :]))
        
        # 计算水平和垂直方向的梯度幅值
        h_grad_magnitude = np.sqrt(h_grad_x**2 + h_grad_y**2)
        v_grad_magnitude = np.sqrt(v_grad_x**2 + v_grad_y**2)
        
        # 计算权重，梯度越小，权重越大
        h_weight = 1.0 / (h_grad_magnitude + 1e-6)
        v_weight = 1.0 / (v_grad_magnitude + 1e-6)
        
        # 归一化权重
        sum_weights = h_weight + v_weight
        h_weight_norm = h_weight / sum_weights
        v_weight_norm = v_weight / sum_weights
        
        # 加权组合两个方向的相位
        combined_phase = horizontal_phase * h_weight_norm + vertical_phase * v_weight_norm
        
        # 进行平滑处理
        self.emit_progress("对组合相位进行平滑处理...")
        smoothed_combined = self.smooth_unwrapped_phase(combined_phase, kernel_size=5)
        
        # 显示组合前后的统计信息
        h_min, h_max = np.min(horizontal_phase), np.max(horizontal_phase)
        v_min, v_max = np.min(vertical_phase), np.max(vertical_phase)
        c_min, c_max = np.min(smoothed_combined), np.max(smoothed_combined)
        
        self.emit_progress(f"水平方向相位范围: [{h_min:.4f}, {h_max:.4f}]")
        self.emit_progress(f"垂直方向相位范围: [{v_min:.4f}, {v_max:.4f}]")
        self.emit_progress(f"组合后相位范围: [{c_min:.4f}, {c_max:.4f}]")
        
        # 保存组合结果
        if not os.path.exists("results"):
            os.makedirs("results")
            
        # 使用英文命名，避免中文乱码
        prefix = "combined_"
        
        # 将相位值缩放到[0,255]范围用于显示和保存
        combined_scaled = np.rint(smoothed_combined*255/(32*math.pi))
        combined_scaled_uint = combined_scaled.astype(np.uint8)
        
        # 应用伪彩色映射以增强可视化效果
        combined_color = cv.applyColorMap(combined_scaled_uint, cv.COLORMAP_JET)
        cv.imwrite(f"results/{prefix}unwrapped_phase_color.png", combined_color)
        
        # 保存为高度图
        combined_norm = smoothed_combined / np.max(smoothed_combined)
        cv.imwrite(f"results/{prefix}unwrapped_phase_height.png", (combined_norm * 65535).astype(np.uint16))
        
        # 保存原始缩放视图
        cv.imwrite(f"results/{prefix}unwrapped_phase_original.png", combined_scaled_uint)
        
        # 应用直方图均衡化
        combined_eq = cv.equalizeHist(combined_scaled_uint)
        cv.imwrite(f"results/{prefix}unwrapped_phase_equalized.png", combined_eq)
        
        self.emit_progress("水平和垂直相位组合完成，结果已保存到results目录")
        return smoothed_combined


# 相位值显示标签，支持鼠标交互
class PhaseImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._highlight_opacity = 0.0
        self.phase_data = None
        self.horizontal_phase = None  # 水平方向相位数据
        self.vertical_phase = None    # 垂直方向相位数据
        self.last_position = None
        self.hover_position = None
        self.highlight_radius = 30  # 高亮效果半径
        self.animation_in_progress = False
        
        # 设置鼠标追踪
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        
        # 设置样式
        self.setStyleSheet("""
            background-color: #000000; 
            border: 1px solid #444444;
            border-radius: 8px;
        """)
        
        # 创建标签和背景渐变
        self.setText("等待相位数据...")
        self.setFont(QFont("Segoe UI", 12))
        
        # 创建点击时的动画效果
        self.click_animation = QPropertyAnimation(self, b"highlight_opacity")
        self.click_animation.setDuration(500)  # 动画持续时间(毫秒)
        self.click_animation.setStartValue(1.0)
        self.click_animation.setEndValue(0.0)
        self.click_animation.setEasingCurve(QEasingCurve.OutQuad)
    
    def get_highlight_opacity(self):
        return self._highlight_opacity
        
    def set_highlight_opacity(self, opacity):
        self._highlight_opacity = opacity
        self.update()  # 触发重绘
        
    # 创建属性，用于动画
    highlight_opacity = Property(float, get_highlight_opacity, set_highlight_opacity)
        
    def setPhaseData(self, phase_data, horizontal_phase=None, vertical_phase=None):
        """设置相位数据，可选设置水平和垂直方向的相位"""
        self.phase_data = phase_data
        self.horizontal_phase = horizontal_phase
        self.vertical_phase = vertical_phase
        
    def mouseMoveEvent(self, event):
        if self.phase_data is not None:
            # 获取鼠标在图像上的位置
            pos = event.position()
            x, y = int(pos.x()), int(pos.y())
            
            # 更新悬停位置
            self.hover_position = (x, y)
            
            # 获取图像尺寸
            if self.pixmap():
                img_w = self.pixmap().width()
                img_h = self.pixmap().height()
                
                # 确保鼠标在图像内
                if 0 <= x < img_w and 0 <= y < img_h:
                    # 计算在原始相位数据中的位置
                    orig_x = int(x * self.phase_data.shape[1] / img_w)
                    orig_y = int(y * self.phase_data.shape[0] / img_h)
                    
                    # 确保在原始数据范围内
                    if 0 <= orig_x < self.phase_data.shape[1] and 0 <= orig_y < self.phase_data.shape[0]:
                        # 获取相位值
                        phase_value = self.phase_data[orig_y, orig_x]
                        # 计算周期数
                        period = phase_value / (2 * math.pi)
                        
                        # 构建状态栏消息
                        status_msg = f"位置: ({orig_x}, {orig_y})  |  相位值: {phase_value:.6f} rad  |  周期数: {period:.6f}"
                        
                        # 如果存在水平和垂直相位数据，则显示这些信息
                        if self.horizontal_phase is not None and self.vertical_phase is not None:
                            if (0 <= orig_x < self.horizontal_phase.shape[1] and 
                                0 <= orig_y < self.horizontal_phase.shape[0] and
                                0 <= orig_x < self.vertical_phase.shape[1] and
                                0 <= orig_y < self.vertical_phase.shape[0]):
                                
                                h_phase = self.horizontal_phase[orig_y, orig_x]
                                v_phase = self.vertical_phase[orig_y, orig_x]
                                h_period = h_phase / (2 * math.pi)
                                v_period = v_phase / (2 * math.pi)
                                
                                status_msg = (f"位置: ({orig_x}, {orig_y})  |  "
                                             f"相位值: {phase_value:.6f} rad  |  周期数: {period:.6f}  |  "
                                             f"水平相位: {h_phase:.6f} rad  |  垂直相位: {v_phase:.6f} rad")
                        
                        # 更新状态栏
                        if self.window().statusBar() is not None:
                            self.window().statusBar().showMessage(status_msg)
                        
            self.update()  # 触发重绘，更新悬停效果
        
        super().mouseMoveEvent(event)
    
    def mousePressEvent(self, event):
        """鼠标按下事件，添加点击动画"""
        if event.button() == Qt.LeftButton and self.phase_data is not None:
            # 获取鼠标在图像上的位置
            pos = event.position()
            x, y = int(pos.x()), int(pos.y())
            
            # 设置标记位置
            self.last_position = (x, y)
            
            # 启动点击动画
            self.click_animation.stop()
            self._highlight_opacity = 1.0
            self.click_animation.start()
            
        super().mousePressEvent(event)
        
    def leaveEvent(self, event):
        """鼠标离开事件"""
        self.hover_position = None
        self.update()
        super().leaveEvent(event)
    
    def paintEvent(self, event):
        """绘制事件，添加交互效果"""
        super().paintEvent(event)
        
        if not self.pixmap():
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制悬停效果
        if self.hover_position and self.phase_data is not None:
            x, y = self.hover_position
            
            # 创建半透明的悬停圆圈
            painter.setPen(Qt.NoPen)
            hover_color = QColor(255, 255, 255, 40)  # 半透明白色
            painter.setBrush(hover_color)
            painter.drawEllipse(x - 15, y - 15, 30, 30)
            
            # 绘制十字准线
            painter.setPen(QPen(QColor(255, 255, 0, 150), 1, Qt.DashLine))
            painter.drawLine(x, 0, x, self.height())
            painter.drawLine(0, y, self.width(), y)
        
        # 绘制选择标记
        if self.last_position and self.phase_data is not None:
            x, y = self.last_position
            
            # 绘制点击高亮效果
            if self._highlight_opacity > 0:
                highlight_color = QColor(52, 152, 219, int(100 * self._highlight_opacity))
                painter.setPen(Qt.NoPen)
                painter.setBrush(highlight_color)
                radius = self.highlight_radius + int(20 * (1 - self._highlight_opacity))
                painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
            
            # 绘制十字标记
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            marker_size = 12
            painter.drawLine(x - marker_size, y, x + marker_size, y)
            painter.drawLine(x, y - marker_size, x, y + marker_size)
            
            # 绘制小圆圈
            painter.setPen(QPen(QColor(255, 255, 0), 1.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(x - 6, y - 6, 12, 12)
        
        painter.end()


# 主窗口
class UnwrappedPhaseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 设置窗口属性
        self.setWindowTitle("解包裹相位分析程序")
        self.setMinimumSize(1200, 800)
        
        # 创建中央部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 创建带有阴影效果的状态栏
        status_bar = self.statusBar()
        status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #F5F7FA;
                color: #34495E;
                border-top: 1px solid #E0E0E0;
                min-height: 24px;
            }
        """)
        status_bar.showMessage("就绪")
        
        # 创建主布局
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(16)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 创建标题标签
        title_label = QLabel("解包裹相位分析")
        title_label.setObjectName("headingLabel")
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.main_layout.addWidget(title_label)
        
        # 创建控制面板
        self.create_control_panel()
        
        # 创建显示区域
        self.create_display_area()
        
        # 创建处理线程实例（初始为None）
        self.processing_thread = None
        
        # 初始化相位数据
        self.unwrapped_phase = None
        self.wrapped_phase = None
        self.k1_matrix = None
        self.k2_matrix = None
        self.fringe_images = None
        self.horizontal_phase = None
        self.vertical_phase = None
        
        # 初始化额外的显示选项，用于组合模式
        self.combined_display_options = []
        
        # 初始化方向选择 - 默认启用所有方向选择选项
        self.direction_mode.setCurrentIndex(0)  # 默认选择水平方向
        # 启用方向选择下拉框
        self.direction_mode.setEnabled(True)
        
        # 确保所有方向选项都是可选的
        for i in range(self.direction_mode.count()):
            if self.direction_mode.model().item(i):
                self.direction_mode.model().item(i).setEnabled(True)
                
        # 初始化提示信息
        self.direction_mode.setToolTip("选择解包裹方向：水平方向使用I1-I4图像，垂直方向使用I5-I8图像，组合模式会综合两种结果")
        
    def create_control_panel(self):
        """创建控制面板"""
        control_group = QGroupBox("控制面板")
        control_group.setStyleSheet("""
            QGroupBox {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                            stop:0 #FFFFFF, stop:1 #F5F7FA);
            }
        """)
        
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(16)
        
        # 文件夹选择部分
        folder_layout = QHBoxLayout()
        folder_layout.setSpacing(12)
        
        folder_label = QLabel("相移图像文件夹:")
        folder_label.setFixedWidth(120)
        
        self.folder_path_label = QLabel("未选择文件夹")
        self.folder_path_label.setObjectName("infoLabel")
        self.folder_path_label.setWordWrap(True)
        
        folder_button = QPushButton("选择文件夹")
        folder_button.setObjectName("secondaryButton")
        folder_button.setIcon(QIcon.fromTheme("folder-open"))
        folder_button.clicked.connect(self.select_folder)
        folder_button.setFixedWidth(120)
        
        folder_layout.addWidget(folder_label)
        folder_layout.addWidget(self.folder_path_label)
        folder_layout.addWidget(folder_button)
        
        control_layout.addLayout(folder_layout)
        
        # 水平分割线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #E0E0E0;")
        separator.setMaximumHeight(1)
        control_layout.addWidget(separator)
        
        # 选项部分
        options_layout = QHBoxLayout()
        options_layout.setSpacing(15)
        
        # 显示模式选择
        display_label = QLabel("显示模式:")
        display_label.setFixedWidth(80)
        
        self.display_mode = QComboBox()
        self.display_mode.addItems([
            "彩色相位图", 
            "灰度相位图", 
            "直方图均衡化相位图", 
            "包裹相位图",
            "K1矩阵图",
            "K2矩阵图"
        ])
        self.display_mode.currentIndexChanged.connect(self.update_display)
        self.display_mode.setFixedWidth(180)
        
        # 添加解包裹方向选择
        direction_label = QLabel("解包裹方向:")
        direction_label.setFixedWidth(80)
        
        self.direction_mode = QComboBox()
        self.direction_mode.addItems([
            "水平方向", 
            "垂直方向", 
            "水平+垂直组合"
        ])
        self.direction_mode.setFixedWidth(120)
        self.direction_mode.setToolTip("水平方向使用I1-I4图像，垂直方向使用I5-I8图像，组合模式会综合两种结果")
        # 确保启用QComboBox，但在select_folder中根据图像数量可能会禁用某些选项
        self.direction_mode.setEnabled(True)
        # 连接信号，当方向选择变化时更新显示选项
        self.direction_mode.currentIndexChanged.connect(self.on_direction_changed)
        
        # 保存中间结果复选框
        self.save_intermediate_cb = QCheckBox("保存中间结果")
        self.save_intermediate_cb.setChecked(True)
        
        # 添加处理按钮
        self.process_button = QPushButton("开始处理")
        self.process_button.setObjectName("primaryButton")
        self.process_button.clicked.connect(self.start_processing)
        self.process_button.setEnabled(False)  # 初始禁用，直到选择文件夹
        self.process_button.setFixedWidth(140)
        
        options_layout.addWidget(display_label)
        options_layout.addWidget(self.display_mode)
        options_layout.addWidget(direction_label)
        options_layout.addWidget(self.direction_mode)
        options_layout.addStretch(1)
        options_layout.addWidget(self.save_intermediate_cb)
        options_layout.addStretch(1)
        options_layout.addWidget(self.process_button)
        
        control_layout.addLayout(options_layout)

        # --- 图像尺寸标准化选项区 ---
        normalization_group = QGroupBox("图像尺寸标准化")
        normalization_group.setStyleSheet("QGroupBox { background: #F9FBFC; border: 1px solid #E0E0E0; border-radius: 8px; margin-top: 8px; padding: 8px; }")
        normalization_layout = QHBoxLayout(normalization_group)
        normalization_layout.setSpacing(10)

        norm_label = QLabel("标准化方式:")
        norm_label.setFixedWidth(90)
        self.norm_method_cb = QComboBox()
        self.norm_method_cb.addItems([
            "自动裁剪到最小尺寸",
            "缩放到相同尺寸",
            "手动指定目标尺寸"
        ])
        self.norm_method_cb.setFixedWidth(150)
        self.norm_method_cb.currentIndexChanged.connect(self.on_norm_method_changed)

        self.norm_width_label = QLabel("宽:")
        self.norm_width_label.setFixedWidth(30)
        self.norm_width_input = QLineEdit()
        self.norm_width_input.setFixedWidth(60)
        self.norm_width_input.setPlaceholderText("宽度")
        self.norm_width_input.setValidator(QIntValidator(1, 10000, self))

        self.norm_height_label = QLabel("高:")
        self.norm_height_label.setFixedWidth(30)
        self.norm_height_input = QLineEdit()
        self.norm_height_input.setFixedWidth(60)
        self.norm_height_input.setPlaceholderText("高度")
        self.norm_height_input.setValidator(QIntValidator(1, 10000, self))

        self.norm_mode_label = QLabel("调整方式:")
        self.norm_mode_label.setFixedWidth(60)
        self.norm_mode_cb = QComboBox()
        self.norm_mode_cb.addItems(["裁剪", "缩放"])
        self.norm_mode_cb.setFixedWidth(60)

        # 仅在手动指定时显示
        self.norm_width_label.hide()
        self.norm_width_input.hide()
        self.norm_height_label.hide()
        self.norm_height_input.hide()
        self.norm_mode_label.hide()
        self.norm_mode_cb.hide()

        normalization_layout.addWidget(norm_label)
        normalization_layout.addWidget(self.norm_method_cb)
        normalization_layout.addWidget(self.norm_width_label)
        normalization_layout.addWidget(self.norm_width_input)
        normalization_layout.addWidget(self.norm_height_label)
        normalization_layout.addWidget(self.norm_height_input)
        normalization_layout.addWidget(self.norm_mode_label)
        normalization_layout.addWidget(self.norm_mode_cb)
        normalization_layout.addStretch(1)

        # 帮助按钮
        help_btn = QToolButton()
        help_btn.setText("?")
        help_btn.setToolTip("关于图像尺寸标准化")
        help_btn.setStyleSheet("QToolButton { font-weight:bold; font-size: 13pt; color: #3498DB; background: #F5F7FA; border-radius: 10px; width: 22px; height: 22px; }")
        help_btn.clicked.connect(self.show_norm_help)
        normalization_layout.addWidget(help_btn)

        control_layout.addWidget(normalization_group)

        # 当前标准化设置信息
        self.norm_info_label = QLabel("当前标准化: 自动裁剪到最小尺寸")
        self.norm_info_label.setObjectName("infoLabel")
        self.norm_info_label.setStyleSheet("font-size:10pt;color:#888;")
        control_layout.addWidget(self.norm_info_label)
        
        # 进度信息区域
        progress_info_layout = QVBoxLayout()
        progress_info_layout.setSpacing(8)

        progress_label = QLabel("处理进度:")

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(12)

        # 进度详细信息
        self.progress_info = QLabel("就绪")
        self.progress_info.setObjectName("infoLabel")
        self.progress_info.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.progress_info.setWordWrap(True)
        self.progress_info.setMinimumHeight(40)
        self.progress_info.setMaximumHeight(60)

        progress_info_layout.addWidget(progress_label)
        progress_info_layout.addWidget(self.progress_bar)
        progress_info_layout.addWidget(self.progress_info)

        control_layout.addLayout(progress_info_layout)
        
        self.main_layout.addWidget(control_group)
        
    def create_display_area(self):
        """创建显示区域"""
        display_layout = QHBoxLayout()
        display_layout.setSpacing(16)
        
        # 创建图像显示容器
        image_container = QGroupBox("相位图像")
        image_layout = QVBoxLayout(image_container)
        
        # 创建图像显示标签
        self.phase_image = PhaseImageLabel()
        
        # 将图像标签添加到滚动区域
        image_scroll = QScrollArea()
        image_scroll.setWidget(self.phase_image)
        image_scroll.setWidgetResizable(True)
        image_scroll.setStyleSheet("border: none; background-color: transparent;")
        
        image_layout.addWidget(image_scroll)
        
        # 创建信息面板
        info_panel = QFrame()
        info_panel.setObjectName("infoPanel")
        info_panel.setFrameShape(QFrame.StyledPanel)
        info_panel.setFrameShadow(QFrame.Raised)
        info_panel.setMaximumWidth(350)
        
        info_layout = QVBoxLayout(info_panel)
        info_layout.setSpacing(16)
        
        # 相位信息
        phase_info_group = QGroupBox("相位数据统计")
        phase_info_layout = QGridLayout(phase_info_group)
        phase_info_layout.setVerticalSpacing(8)
        phase_info_layout.setHorizontalSpacing(12)
        
        self.info_labels = {
            "unwrapped_min": QLabel("最小值: -"),
            "unwrapped_max": QLabel("最大值: -"),
            "unwrapped_mean": QLabel("平均值: -"),
            "wrapped_min": QLabel("包裹相位最小值: -"),
            "wrapped_max": QLabel("包裹相位最大值: -"),
            "image_size": QLabel("图像尺寸: -"),
            "k1_range": QLabel("K1范围: -"),
            "k2_range": QLabel("K2范围: -")
        }
        
        row = 0
        for key, label in self.info_labels.items():
            phase_info_layout.addWidget(label, row, 0)
            row += 1
            
        info_layout.addWidget(phase_info_group)
        
        # 文件信息
        file_info_group = QGroupBox("文件信息")
        file_info_layout = QVBoxLayout(file_info_group)
        file_info_layout.setSpacing(10)
        
        self.fringe_file_list = QLabel("未加载相移图像")
        self.fringe_file_list.setWordWrap(True)
        self.fringe_file_list.setObjectName("infoLabel")
        file_info_layout.addWidget(self.fringe_file_list)
        
        self.result_file_list = QLabel("未生成结果")
        self.result_file_list.setWordWrap(True)
        self.result_file_list.setObjectName("infoLabel")
        file_info_layout.addWidget(self.result_file_list)
        
        info_layout.addWidget(file_info_group)
        
        # 操作按钮
        button_layout = QHBoxLayout()
        
        save_button = QPushButton("保存当前视图")
        save_button.setObjectName("secondaryButton")
        save_button.clicked.connect(self.save_current_view)
        button_layout.addWidget(save_button)
        button_layout.addStretch()
        
        info_layout.addLayout(button_layout)
        info_layout.addStretch()
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setChildrenCollapsible(False)
        
        splitter.addWidget(image_container)
        splitter.addWidget(info_panel)
        
        # 设置分割器比例
        splitter.setSizes([800, 350])
        
        display_layout.addWidget(splitter)
        self.main_layout.addLayout(display_layout, 1)  # 1是伸展因子，让显示区域占据更多空间
        
    def select_folder(self):
        """选择四步相移图像文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择四步相移图像文件夹")
        if folder_path:
            self.folder_path_label.setText(folder_path)
            
            # 检查文件夹中是否有足够的图像文件
            image_files = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
            
            # 获取当前选择的方向
            current_direction = self.direction_mode.currentText()
            
            if len(image_files) < 4:
                # 如果图像数量不足4个，显示警告但不禁用方向选择
                QMessageBox.warning(self, "警告", f"文件夹中只有{len(image_files)}个图像文件，需要至少4个才能进行解包裹")
                self.process_button.setEnabled(False)
                self.fringe_file_list.setText(f"图像数量不足: 找到{len(image_files)}个图像文件，至少需要4个")
            else:
                self.process_button.setEnabled(True)
                
                # 更新图像文件列表显示
                sorted_files = sorted(image_files)
                
                # 分析文件数量
                if len(sorted_files) >= 8:
                    # 有足够图像支持所有方向
                    h_files = sorted_files[:4]
                    v_files = sorted_files[4:8]
                    self.fringe_file_list.setText(f"水平方向图像: {', '.join(h_files)}\n垂直方向图像: {', '.join(v_files)}")
                else:
                    # 只有足够支持水平方向
                    self.fringe_file_list.setText(f"找到图像文件: {', '.join(sorted_files[:4])}\n注意: 需要至少8个图像才能使用垂直方向或组合模式")
                    
                    # 检查当前选择的方向是否需要8个图像
                    if (current_direction == "垂直方向" or current_direction == "水平+垂直组合") and len(sorted_files) < 8:
                        # 显示警告，但不强制改变选择
                        QMessageBox.warning(self, "方向选择提示", 
                                          f"您选择了{current_direction}，但文件夹中只有{len(sorted_files)}个图像文件，" +
                                          "垂直方向和组合模式需要至少8个图像文件。\n\n" +
                                          "处理时可能会出现错误，建议选择水平方向或提供更多图像。")
                    
                # 根据当前选择的方向更新显示选项
                self.on_direction_changed()
                    
                # 重置进度条
                self.progress_bar.setValue(0)
                self.progress_info.setText("就绪 - 点击\"开始处理\"按钮开始分析")
    
    def start_processing(self):
        """开始处理"""
        if not self.folder_path_label.text() or self.folder_path_label.text() == "未选择文件夹":
            QMessageBox.warning(self, "警告", "请先选择四步相移图像文件夹")
            return
            
        # 禁用处理按钮，避免重复点击
        self.process_button.setEnabled(False)
        self.progress_info.setText("正在处理...")
        self.progress_bar.setValue(5)  # 初始进度

        # 获取标准化参数
        normalization_params = self.get_normalization_params()
        self.update_norm_info()  # 确保信息区同步
        
        # 获取选择的解包裹方向
        direction_mode = self.direction_mode.currentText()
        if direction_mode == "水平方向":
            direction = "horizontal"
        elif direction_mode == "垂直方向":
            direction = "vertical"
        else:  # 水平+垂直组合
            direction = "combined"
            
        # 显示当前模式
        self.progress_info.setText(f"使用{direction_mode}解包裹模式，正在处理...")

        # 创建并启动处理线程
        self.processing_thread = ProcessingThread(
            self.folder_path_label.text(), 
            self.save_intermediate_cb.isChecked(),
            normalization_params=normalization_params,
            direction=direction
        )
        
        # 连接信号
        self.processing_thread.progress_signal.connect(self.update_progress)
        self.processing_thread.result_signal.connect(self.process_completed)
        self.processing_thread.wrapped_signal.connect(self.set_wrapped_phase)
        self.processing_thread.k1_signal.connect(self.set_k1_matrix)
        self.processing_thread.k2_signal.connect(self.set_k2_matrix)
        self.processing_thread.fringe_signal.connect(self.set_fringe_images)
        self.processing_thread.horizontal_phase_signal.connect(self.set_horizontal_phase)
        self.processing_thread.vertical_phase_signal.connect(self.set_vertical_phase)
        self.processing_thread.finished_signal.connect(self.processing_finished)
        self.processing_thread.error_signal.connect(self.processing_error)
        
        # 启动线程
        self.processing_thread.start()
        
    def update_progress(self, message):
        """更新进度信息"""
        # 拼接标准化信息
        norm_params = self.get_normalization_params()
        if norm_params["target_size"]:
            norm_info = f"标准化: {('缩放' if norm_params['method']=='resize' else '裁剪')}到{norm_params['target_size'][1]}x{norm_params['target_size'][0]}"
        else:
            norm_info = f"标准化: {'缩放' if norm_params['method']=='resize' else '裁剪'}到最小尺寸"
        self.progress_info.setText(f"{message}\n{norm_info}")
        self.statusBar().showMessage(message)
        
        # 根据处理阶段更新进度条
        if "读取四步相移图像" in message:
            self.progress_bar.setValue(10)
        elif "计算包裹相位" in message:
            self.progress_bar.setValue(30)
        elif "解码格雷码" in message:
            self.progress_bar.setValue(50)
        elif "计算解包裹相位" in message:
            self.progress_bar.setValue(70)
        elif "保存结果" in message:
            self.progress_bar.setValue(90)
        elif "处理完成" in message:
            self.progress_bar.setValue(100)
        
    def process_completed(self, unwrapped_phase):
        """处理完成"""
        self.unwrapped_phase = unwrapped_phase
        
        # 更新信息标签
        self.info_labels["unwrapped_min"].setText(f"最小值: {np.min(unwrapped_phase):.6f} rad")
        self.info_labels["unwrapped_max"].setText(f"最大值: {np.max(unwrapped_phase):.6f} rad")
        self.info_labels["unwrapped_mean"].setText(f"平均值: {np.mean(unwrapped_phase):.6f} rad")
        self.info_labels["image_size"].setText(f"图像尺寸: {unwrapped_phase.shape[1]}x{unwrapped_phase.shape[0]}")

        # 显示标准化尺寸
        norm_params = self.get_normalization_params()
        if norm_params["target_size"]:
            norm_str = f"标准化尺寸: {norm_params['target_size'][1]}x{norm_params['target_size'][0]}"
        else:
            norm_str = f"标准化尺寸: {unwrapped_phase.shape[1]}x{unwrapped_phase.shape[0]} (最小公共尺寸)"
        self.info_labels["image_size"].setText(self.info_labels["image_size"].text() + f"\n{norm_str}")
        
        # 更新结果文件列表
        if os.path.exists("results"):
            result_files = [f for f in os.listdir("results") 
                          if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))]
            self.result_file_list.setText(f"生成结果: {', '.join(result_files)}")
        
        # 检查当前的方向模式
        current_direction = self.direction_mode.currentText()
        
        # 如果是组合模式，确保显示选项中包含水平和垂直相位图选项
        if current_direction == "水平+垂直组合" and self.horizontal_phase is not None and self.vertical_phase is not None:
            # 手动触发方向变化处理，以确保显示选项正确
            self.on_direction_changed()
        
        # 更新显示
        self.update_display()
        
    def set_wrapped_phase(self, wrapped_phase):
        """设置包裹相位数据"""
        self.wrapped_phase = wrapped_phase
        
        # 更新信息标签
        self.info_labels["wrapped_min"].setText(f"包裹相位最小值: {np.min(wrapped_phase):.6f} rad")
        self.info_labels["wrapped_max"].setText(f"包裹相位最大值: {np.max(wrapped_phase):.6f} rad")
        
    def set_k1_matrix(self, k1):
        """设置K1矩阵数据"""
        self.k1_matrix = k1
        self.info_labels["k1_range"].setText(f"K1范围: [{np.min(k1)}, {np.max(k1)}]")
        
    def set_k2_matrix(self, k2):
        """设置K2矩阵数据"""
        self.k2_matrix = k2
        self.info_labels["k2_range"].setText(f"K2范围: [{np.min(k2)}, {np.max(k2)}]")
        
    def set_fringe_images(self, fringe_images):
        """设置四步相移图像"""
        self.fringe_images = fringe_images
        
    def set_horizontal_phase(self, horizontal_phase):
        """设置水平方向解包裹相位"""
        self.horizontal_phase = horizontal_phase
        
    def set_vertical_phase(self, vertical_phase):
        """设置垂直方向解包裹相位"""
        self.vertical_phase = vertical_phase
        
    def processing_finished(self):
        """处理完成后的操作"""
        self.process_button.setEnabled(True)
        self.statusBar().showMessage("处理完成")
        self.progress_bar.setValue(100)
        self.progress_info.setText("处理完成 - 可以查看结果并保存")
        
    def processing_error(self, error_message):
        """处理错误"""
        self.process_button.setEnabled(True)
        self.progress_info.setText(f"错误: {error_message}")
        self.statusBar().showMessage("处理出错")
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "错误", error_message)
        
    def update_display(self):
        """更新显示内容"""
        current_mode = self.display_mode.currentText()
        
        # 标准显示模式
        if current_mode == "彩色相位图" and self.unwrapped_phase is not None:
            # 使用伪彩色映射
            upha_scaled = np.rint(self.unwrapped_phase*255/(32*math.pi))
            upha_scaled_uint = upha_scaled.astype(np.uint8)
            upha_color = cv.applyColorMap(upha_scaled_uint, cv.COLORMAP_JET)
            
            # 如果是组合模式的结果，则传递水平和垂直相位数据
            if self.horizontal_phase is not None and self.vertical_phase is not None:
                self.display_image(upha_color, self.unwrapped_phase, 
                                  self.horizontal_phase, self.vertical_phase)
            else:
                self.display_image(upha_color, self.unwrapped_phase)
            
        elif current_mode == "灰度相位图" and self.unwrapped_phase is not None:
            # 灰度显示
            upha_scaled = np.rint(self.unwrapped_phase*255/(32*math.pi))
            upha_scaled_uint = upha_scaled.astype(np.uint8)
            
            # 如果是组合模式的结果，则传递水平和垂直相位数据
            if self.horizontal_phase is not None and self.vertical_phase is not None:
                self.display_image(upha_scaled_uint, self.unwrapped_phase, 
                                  self.horizontal_phase, self.vertical_phase)
            else:
                self.display_image(upha_scaled_uint, self.unwrapped_phase)
            
        elif current_mode == "直方图均衡化相位图" and self.unwrapped_phase is not None:
            # 直方图均衡化
            upha_scaled = np.rint(self.unwrapped_phase*255/(32*math.pi))
            upha_scaled_uint = upha_scaled.astype(np.uint8)
            upha_eq = cv.equalizeHist(upha_scaled_uint)
            
            # 如果是组合模式的结果，则传递水平和垂直相位数据
            if self.horizontal_phase is not None and self.vertical_phase is not None:
                self.display_image(upha_eq, self.unwrapped_phase, 
                                  self.horizontal_phase, self.vertical_phase)
            else:
                self.display_image(upha_eq, self.unwrapped_phase)
            
        elif current_mode == "包裹相位图" and self.wrapped_phase is not None:
            # 包裹相位显示
            wrapped_scaled = np.rint(self.wrapped_phase*255/(2*math.pi))
            wrapped_uint = wrapped_scaled.astype(np.uint8)
            wrapped_color = cv.applyColorMap(wrapped_uint, cv.COLORMAP_JET)
            self.display_image(wrapped_color, self.wrapped_phase)
            
        elif current_mode == "K1矩阵图" and self.k1_matrix is not None:
            # K1矩阵显示
            k1_scaled = (self.k1_matrix * (255/15)).astype(np.uint8)
            k1_color = cv.applyColorMap(k1_scaled, cv.COLORMAP_JET)
            self.display_image(k1_color, self.k1_matrix)
            
        elif current_mode == "K2矩阵图" and self.k2_matrix is not None:
            # K2矩阵显示
            k2_scaled = (self.k2_matrix * (255/31)).astype(np.uint8)
            k2_color = cv.applyColorMap(k2_scaled, cv.COLORMAP_JET)
            self.display_image(k2_color, self.k2_matrix)
        
        # 组合模式特有的显示选项
        elif current_mode == "水平方向解包裹相位图" and self.horizontal_phase is not None:
            # 显示水平方向解包裹相位图
            h_scaled = np.rint(self.horizontal_phase*255/(32*math.pi))
            h_scaled_uint = h_scaled.astype(np.uint8)
            h_color = cv.applyColorMap(h_scaled_uint, cv.COLORMAP_JET)
            self.display_image(h_color, self.horizontal_phase)
            
        elif current_mode == "垂直方向解包裹相位图" and self.vertical_phase is not None:
            # 显示垂直方向解包裹相位图
            v_scaled = np.rint(self.vertical_phase*255/(32*math.pi))
            v_scaled_uint = v_scaled.astype(np.uint8)
            v_color = cv.applyColorMap(v_scaled_uint, cv.COLORMAP_JET)
            self.display_image(v_color, self.vertical_phase)
            
        else:
            self.phase_image.setText("无可显示的数据")
            self.phase_image.setPhaseData(None)
            
    def display_image(self, cv_img, phase_data=None, horizontal_phase=None, vertical_phase=None):
        """将OpenCV图像显示在界面上"""
        if len(cv_img.shape) == 2:  # 灰度图像
            h, w = cv_img.shape
            bytes_per_line = w
            q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:  # 彩色图像
            h, w, c = cv_img.shape
            bytes_per_line = 3 * w
            cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
            q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
        pixmap = QPixmap.fromImage(q_img)
        
        # 获取显示区域的尺寸
        label_width = self.phase_image.width()
        label_height = self.phase_image.height()
        
        # 计算缩放比例，保持宽高比
        width_ratio = label_width / w
        height_ratio = label_height / h
        
        # 使用较小的比例，确保图像完全可见
        scale_ratio = min(width_ratio, height_ratio)
        
        # 如果图像比显示区域小，则放大图像
        if scale_ratio < 0.95:  # 允许5%的边距
            # 缩放到接近但略小于显示区域的尺寸
            scaled_pixmap = pixmap.scaled(
                int(w * scale_ratio * 0.95), 
                int(h * scale_ratio * 0.95),
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
        else:
            # 如果图像比显示区域大，则适当缩小
            scaled_pixmap = pixmap.scaled(
                label_width - 20,  # 留出一些边距
                label_height - 20,
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
        
        self.phase_image.setPixmap(scaled_pixmap)
        self.phase_image.setPhaseData(phase_data, horizontal_phase, vertical_phase)
        
    def save_current_view(self):
        """保存当前视图"""
        if self.phase_image.pixmap() is None:
            QMessageBox.warning(self, "警告", "没有可保存的图像")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "", "图像文件 (*.png *.jpg *.bmp)"
        )
        
        if file_path:
            self.phase_image.pixmap().save(file_path)
            QMessageBox.information(self, "保存成功", f"图像已保存至: {file_path}")

    def on_norm_method_changed(self):
        method = self.norm_method_cb.currentText()
        if method == "手动指定目标尺寸":
            self.norm_width_label.show()
            self.norm_width_input.show()
            self.norm_height_label.show()
            self.norm_height_input.show()
            self.norm_mode_label.show()
            self.norm_mode_cb.show()
        else:
            self.norm_width_label.hide()
            self.norm_width_input.hide()
            self.norm_height_label.hide()
            self.norm_height_input.hide()
            self.norm_mode_label.hide()
            self.norm_mode_cb.hide()
        # 更新信息
        self.update_norm_info()

    def update_norm_info(self):
        method = self.norm_method_cb.currentText()
        if method == "自动裁剪到最小尺寸":
            info = "当前标准化: 自动裁剪到最小尺寸"
        elif method == "缩放到相同尺寸":
            info = "当前标准化: 缩放到所有图像的最小尺寸"
        else:
            w = self.norm_width_input.text() or "-"
            h = self.norm_height_input.text() or "-"
            mode = self.norm_mode_cb.currentText()
            info = f"当前标准化: 手动指定 {w}x{h} ({mode})"
        self.norm_info_label.setText(info)

    def get_normalization_params(self):
        method = self.norm_method_cb.currentText()
        if method == "自动裁剪到最小尺寸":
            return {"method": "crop", "target_size": None}
        elif method == "缩放到相同尺寸":
            return {"method": "resize", "target_size": None}
        else:
            try:
                w = int(self.norm_width_input.text())
                h = int(self.norm_height_input.text())
                mode = self.norm_mode_cb.currentText()
                return {"method": "resize" if mode == "缩放" else "crop", "target_size": (h, w)}
            except Exception:
                return {"method": "crop", "target_size": None}

    def show_norm_help(self):
        msg = (
            "<b>图像尺寸标准化说明</b><br>"
            "<ul>"
            "<li><b>自动裁剪到最小尺寸</b>：所有输入图像将被裁剪为它们的最小公共尺寸，保证像素精确性，可能丢失边缘。</li>"
            "<li><b>缩放到相同尺寸</b>：所有输入图像将被缩放到最小公共尺寸，保留全部内容，可能有插值误差。</li>"
            "<li><b>手动指定目标尺寸</b>：可自定义宽高和调整方式（裁剪/缩放）。</li>"
            "</ul>"
            "<span style='color:#888'>标准化有助于避免因分辨率不一致导致的处理错误。建议优先使用自动裁剪，特殊需求时可手动指定。</span>"
        )
        QMessageBox.information(self, "图像尺寸标准化帮助", msg)

    def on_direction_changed(self):
        """当方向选择改变时处理"""
        current_direction = self.direction_mode.currentText()
        
        # 获取当前选定的显示模式
        current_display = self.display_mode.currentText()
        
        # 如果选择了组合模式，添加水平和垂直相位图显示选项
        if current_direction == "水平+垂直组合":
            # 先检查是否已经添加了这些选项
            has_horizontal = False
            has_vertical = False
            
            # 检查显示模式中是否已包含水平和垂直相位图选项
            for i in range(self.display_mode.count()):
                text = self.display_mode.itemText(i)
                if text == "水平方向解包裹相位图":
                    has_horizontal = True
                elif text == "垂直方向解包裹相位图":
                    has_vertical = True
            
            # 添加缺失的选项
            if not has_horizontal:
                self.display_mode.addItem("水平方向解包裹相位图")
            if not has_vertical:
                self.display_mode.addItem("垂直方向解包裹相位图")
        else:
            # 非组合模式下，移除水平和垂直相位图显示选项
            for text in ["水平方向解包裹相位图", "垂直方向解包裹相位图"]:
                idx = self.display_mode.findText(text)
                if idx >= 0:
                    self.display_mode.removeItem(idx)
                    
        # 尝试保持之前的选择，如果该选项仍然存在
        idx = self.display_mode.findText(current_display)
        if idx >= 0:
            self.display_mode.setCurrentIndex(idx)
        else:
            # 如果之前的选项不存在，则默认选择彩色相位图
            self.display_mode.setCurrentIndex(0)


# UI设计优化总结：
# 1. 设计风格：采用清新主义设计风格，以蓝色(#3498DB)为主色调，配合白色与浅灰色背景
# 2. 视觉层次：通过卡片式设计和阴影效果增强界面层次感
# 3. 交互体验：添加鼠标悬停效果、点击动画、高亮效果等微交互
# 4. 布局优化：优化布局间距，添加分隔线，改善信息展示的逻辑性
# 5. 可读性增强：使用统一的字体和颜色系统，提高信息对比度
# 6. 进度反馈：增加进度条和状态提示，增强用户体验

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 应用全局样式
    app = set_app_style(app)
    
    # 创建并显示主窗口
    window = UnwrappedPhaseApp()
    window.show()
    
    sys.exit(app.exec()) 