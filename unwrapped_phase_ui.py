import sys
import os
import numpy as np
import cv2 as cv
import math
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QFileDialog, QComboBox, QGroupBox, QSplitter,
                             QCheckBox, QScrollArea, QFrame, QMessageBox, QGridLayout, QStatusBar,
                             QProgressBar)
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QIcon, QFontDatabase
from PySide6.QtCore import Qt, QThread, Signal, QSize, QPropertyAnimation, QEasingCurve, Property

# 导入解相位算法
from unwrapped_phase_standalone import UnwrappedPhase, WrappedPhase, GrayCode

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
    finished_signal = Signal()  # 完成信号
    error_signal = Signal(str)  # 错误信号
    
    def __init__(self, fringe_folder, save_intermediate=True):
        super().__init__()
        self.fringe_folder = fringe_folder
        self.save_intermediate = save_intermediate
        
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
            
            # 创建解包裹相位计算器实例
            self.progress_signal.emit("初始化解包裹相位计算器...")
            unwrapper = CustomUnwrappedPhase(self.progress_signal)
            
            # 获取四步相移图像
            self.progress_signal.emit("读取四步相移图像...")
            fringe_images = unwrapper.get_fringe_images()
            self.fringe_signal.emit(fringe_images)
            
            # 计算包裹相位
            self.progress_signal.emit("计算包裹相位...")
            wrapped_phase = unwrapper.compute_wrapped_phase(fringe_images)
            self.wrapped_signal.emit(wrapped_phase)
            
            # 获取k1和k2矩阵
            self.progress_signal.emit("解码格雷码...")
            k1, k2 = unwrapper.get_k1_k2()
            self.k1_signal.emit(k1)
            self.k2_signal.emit(k2)
            
            # 计算解包裹相位
            self.progress_signal.emit("计算解包裹相位...")
            unwrapped_phase = unwrapper.compute_unwrapped_phase(wrapped_phase, k1, k2, self.save_intermediate)
            
            # 发送结果
            self.result_signal.emit(unwrapped_phase)
            self.progress_signal.emit("处理完成")
            self.finished_signal.emit()
            
        except Exception as e:
            import traceback
            self.error_signal.emit(f"处理过程中发生错误: {str(e)}\n{traceback.format_exc()}")


# 自定义的解包裹相位类，添加进度信号
class CustomUnwrappedPhase(UnwrappedPhase):
    def __init__(self, progress_signal=None):
        super().__init__()
        self.progress_signal = progress_signal
        
    def emit_progress(self, message):
        if self.progress_signal:
            self.progress_signal.emit(message)
            
    def get_fringe_images(self):
        """获取四步相移图像"""
        self.emit_progress("读取四步相移图像...")
        
        fringe_folder = os.environ.get("FRINGE_FOLDER", "fringe_patterns")
        I = []
        
        # 查找文件夹中的所有图像文件
        image_files = [f for f in os.listdir(fringe_folder) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        
        # 确保至少有4个图像文件
        if len(image_files) < 4:
            self.emit_progress(f"警告: 在{fringe_folder}中只找到{len(image_files)}个图像文件，需要至少4个")
            raise ValueError(f"文件夹{fringe_folder}中的图像文件数量不足，需要至少4个")
        
        # 选择前4个图像文件
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
        
    def compute_unwrapped_phase(self, wrapped_pha, k1, k2, save_intermediate=True):
        """计算解包裹相位"""
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
        
        if not save_intermediate:
            return unwrapped_pha
            
        # 保存结果
        self.emit_progress("保存结果...")
        
        # 将相位值缩放到[0,255]范围用于显示和保存
        # 对于5位格雷码，相位范围约为[0, 32π]
        upha_scaled = np.rint(unwrapped_pha*255/(32*math.pi))
        upha_scaled_uint = upha_scaled.astype(np.uint8)
        
        # 保存不同视图的解包裹相位图像
        cv.imwrite("results/unwrapped_phase_original.png", upha_scaled_uint)
        
        # 应用伪彩色映射以增强可视化效果
        upha_color = cv.applyColorMap(upha_scaled_uint, cv.COLORMAP_JET)
        cv.imwrite("results/unwrapped_phase_color.png", upha_color)
        
        # 应用直方图均衡化以增强对比度
        upha_eq = cv.equalizeHist(upha_scaled_uint)
        cv.imwrite("results/unwrapped_phase_equalized.png", upha_eq)
        
        # 3D可视化（保存为高度图）
        # 将相位值归一化到[0,1]范围
        upha_norm = unwrapped_pha / np.max(unwrapped_pha)
        # 保存为16位PNG，以保留更多细节
        cv.imwrite("results/unwrapped_phase_height.png", (upha_norm * 65535).astype(np.uint16))
        
        self.emit_progress("解包裹相位计算完成")
        return unwrapped_pha


# 相位值显示标签，支持鼠标交互
class PhaseImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.phase_data = None
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
        self._highlight_opacity = 0.0
        
    def get_highlight_opacity(self):
        return self._highlight_opacity
        
    def set_highlight_opacity(self, opacity):
        self._highlight_opacity = opacity
        self.update()  # 触发重绘
        
    # 创建属性，用于动画
    highlight_opacity = Property(float, get_highlight_opacity, set_highlight_opacity)
        
    def setPhaseData(self, phase_data):
        self.phase_data = phase_data
        
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
                        
                        # 更新状态栏
                        if self.window().statusBar() is not None:
                            self.window().statusBar().showMessage(
                                f"位置: ({orig_x}, {orig_y})  |  相位值: {phase_value:.6f} rad  |  周期数: {period:.6f}")
                        
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
        options_layout.addStretch(1)
        options_layout.addWidget(self.save_intermediate_cb)
        options_layout.addStretch(1)
        options_layout.addWidget(self.process_button)
        
        control_layout.addLayout(options_layout)
        
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
            
            if len(image_files) < 4:
                QMessageBox.warning(self, "警告", f"文件夹中只有{len(image_files)}个图像文件，需要至少4个")
                self.process_button.setEnabled(False)
            else:
                self.process_button.setEnabled(True)
                self.fringe_file_list.setText(f"找到图像文件: {', '.join(sorted(image_files)[:4])}")
                
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
        
        # 创建并启动处理线程
        self.processing_thread = ProcessingThread(
            self.folder_path_label.text(), 
            self.save_intermediate_cb.isChecked()
        )
        
        # 连接信号
        self.processing_thread.progress_signal.connect(self.update_progress)
        self.processing_thread.result_signal.connect(self.process_completed)
        self.processing_thread.wrapped_signal.connect(self.set_wrapped_phase)
        self.processing_thread.k1_signal.connect(self.set_k1_matrix)
        self.processing_thread.k2_signal.connect(self.set_k2_matrix)
        self.processing_thread.fringe_signal.connect(self.set_fringe_images)
        self.processing_thread.finished_signal.connect(self.processing_finished)
        self.processing_thread.error_signal.connect(self.processing_error)
        
        # 启动线程
        self.processing_thread.start()
        
    def update_progress(self, message):
        """更新进度信息"""
        self.progress_info.setText(message)
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
        
        # 更新结果文件列表
        if os.path.exists("results"):
            result_files = [f for f in os.listdir("results") 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
            self.result_file_list.setText(f"生成结果: {', '.join(result_files)}")
        
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
        
        if current_mode == "彩色相位图" and self.unwrapped_phase is not None:
            # 使用伪彩色映射
            upha_scaled = np.rint(self.unwrapped_phase*255/(32*math.pi))
            upha_scaled_uint = upha_scaled.astype(np.uint8)
            upha_color = cv.applyColorMap(upha_scaled_uint, cv.COLORMAP_JET)
            self.display_image(upha_color, self.unwrapped_phase)
            
        elif current_mode == "灰度相位图" and self.unwrapped_phase is not None:
            # 灰度显示
            upha_scaled = np.rint(self.unwrapped_phase*255/(32*math.pi))
            upha_scaled_uint = upha_scaled.astype(np.uint8)
            self.display_image(upha_scaled_uint, self.unwrapped_phase)
            
        elif current_mode == "直方图均衡化相位图" and self.unwrapped_phase is not None:
            # 直方图均衡化
            upha_scaled = np.rint(self.unwrapped_phase*255/(32*math.pi))
            upha_scaled_uint = upha_scaled.astype(np.uint8)
            upha_eq = cv.equalizeHist(upha_scaled_uint)
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
            
        else:
            self.phase_image.setText("无可显示的数据")
            self.phase_image.setPhaseData(None)
            
    def display_image(self, cv_img, phase_data=None):
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
        self.phase_image.setPixmap(pixmap)
        self.phase_image.setPhaseData(phase_data)
        
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