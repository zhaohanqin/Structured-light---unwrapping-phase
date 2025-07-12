import sys
import os
import numpy as np
import math
import cv2 as cv
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QRadioButton, 
                             QPushButton, QSpinBox, QGroupBox, QButtonGroup, 
                             QSlider, QFileDialog, QMessageBox, QFormLayout,
                             QComboBox)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QSize
from PySide6.QtGui import QPixmap, QImage, QColor, QPalette, QFont

class FringePatternGenerator(QMainWindow):
    """N步相移条纹图案生成器的UI界面"""
    
    def __init__(self):
        super().__init__()
        
        # 设置窗口标题和大小
        self.setWindowTitle("N步相移条纹图案生成器")
        self.resize(900, 700)
        
        # 设置浅色系主题
        self.setup_theme()
        
        # 初始化UI组件
        self.setup_ui()
        
        # 连接信号和槽
        self.connect_signals()
    
    def setup_theme(self):
        """设置应用浅色系主题"""
        palette = QPalette()
        
        # 设置浅色背景
        palette.setColor(QPalette.Window, QColor(240, 240, 245))
        palette.setColor(QPalette.WindowText, QColor(70, 70, 80))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(245, 245, 250))
        palette.setColor(QPalette.Button, QColor(230, 230, 235))
        palette.setColor(QPalette.ButtonText, QColor(70, 70, 80))
        
        # 设置浅色强调色
        palette.setColor(QPalette.Highlight, QColor(120, 170, 220))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        
        self.setPalette(palette)
        
        # 设置字体
        font = QFont("Arial", 10)
        QApplication.setFont(font)
    
    def setup_ui(self):
        """设置UI界面"""
        # 主窗口中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # 左侧设置面板
        settings_panel = QWidget()
        settings_layout = QVBoxLayout(settings_panel)
        settings_layout.setSpacing(20)
        
        # 右侧预览面板
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setSpacing(15)
        
        # 添加左右两个面板到主布局
        main_layout.addWidget(settings_panel, 1)  # 设置面板
        main_layout.addWidget(preview_panel, 1)   # 预览面板
        
        # 0. 模式选择组
        mode_group = QGroupBox("生成模式")
        mode_layout = QHBoxLayout(mode_group)
        self.mode_group = QButtonGroup(self)
        self.single_mode_radio = QRadioButton("单个方向")
        self.batch_mode_radio = QRadioButton("一键生成 (双方向)")
        self.single_mode_radio.setToolTip("只生成指定方向（垂直或水平）的相移条纹图。")
        self.batch_mode_radio.setToolTip("一次性生成垂直和水平两个方向的相移条纹图。\n垂直条纹命名为 I1, I2...；水平条纹命名为 I(N+1), I(N+2)...")
        self.mode_group.addButton(self.single_mode_radio, 0)
        self.mode_group.addButton(self.batch_mode_radio, 1)
        self.single_mode_radio.setChecked(True)
        mode_layout.addWidget(self.single_mode_radio)
        mode_layout.addWidget(self.batch_mode_radio)
        
        # 1. 条纹方向选择组
        self.direction_group_box = QGroupBox("条纹方向")
        direction_layout = QVBoxLayout(self.direction_group_box)
        direction_layout.setSpacing(10)
        
        # 条纹方向选择按钮
        self.vertical_radio = QRadioButton("垂直条纹 (用于水平方向解包裹)")
        self.horizontal_radio = QRadioButton("水平条纹 (用于垂直方向解包裹)")
        self.vertical_radio.setToolTip("生成的条纹是垂直的，其相位沿水平方向变化，用于水平解包裹。")
        self.horizontal_radio.setToolTip("生成的条纹是水平的，其相位沿垂直方向变化，用于垂直解包裹。")
        self.vertical_radio.setChecked(True)  # 默认选择垂直条纹
        
        # 方向图示
        direction_info = QLabel()
        direction_info.setText("垂直条纹 (┃┃┃┃) 相位沿水平方向变化\n水平条纹 (━━━━) 相位沿垂直方向变化")
        direction_info.setStyleSheet("background-color: #f5f5f7; padding: 8px; border-radius: 4px;")
        
        direction_layout.addWidget(self.vertical_radio)
        direction_layout.addWidget(self.horizontal_radio)
        direction_layout.addWidget(direction_info)
        
        # 2. 图像参数设置组
        params_group = QGroupBox("图像参数")
        params_layout = QFormLayout(params_group)
        params_layout.setSpacing(10)
        
        # 参数说明标签
        params_info = QLabel("调整这些参数可以优化相位重建质量。点击\"参数指南\"获取详细建议。")
        params_info.setStyleSheet("color: #555; font-style: italic; padding: 4px;")
        params_info.setWordWrap(True)
        params_layout.addRow(params_info)
        
        # 相移步数
        self.steps_input = QSpinBox()
        self.steps_input.setRange(3, 100)
        self.steps_input.setValue(4)
        self.steps_input.setToolTip("设置相移的步数 (N)。\n- 步数越多，相位精度越高，但需要采集的图像也越多。\n- 常用值为3, 4, 8。")
        params_layout.addRow("相移步数 (N):", self.steps_input)
        
        # 图像宽度
        self.width_input = QSpinBox()
        self.width_input.setRange(100, 4096)
        self.width_input.setValue(1024)
        self.width_input.setSingleStep(10)
        self.width_input.setToolTip("设置生成图像的宽度（像素）。\n应与您的投影仪或相机分辨率匹配。")
        params_layout.addRow("图像宽度:", self.width_input)
        
        # 图像高度
        self.height_input = QSpinBox()
        self.height_input.setRange(100, 4096)
        self.height_input.setValue(768)
        self.height_input.setSingleStep(10)
        self.height_input.setToolTip("设置生成图像的高度（像素）。\n应与您的投影仪或相机分辨率匹配。")
        params_layout.addRow("图像高度:", self.height_input)
        
        # 条纹频率
        self.freq_layout = QHBoxLayout()
        self.freq_input = QSpinBox()
        self.freq_input.setRange(1, 100)
        self.freq_input.setValue(15)
        self.freq_input.setToolTip("条纹频率影响相位分辨率：\n- 高频率(20-30)：提供更高的3D重建精度，但增加解包裹错误风险\n- 低频率(8-15)：更稳健，但3D分辨率较低\n- 推荐值：初始扫描使用8-15，精细扫描使用20-30")
        self.freq_slider = QSlider(Qt.Horizontal)
        self.freq_slider.setRange(1, 100)
        self.freq_slider.setValue(15)
        self.freq_slider.setToolTip(self.freq_input.toolTip())
        self.freq_layout.addWidget(self.freq_input)
        self.freq_layout.addWidget(self.freq_slider)
        params_layout.addRow("条纹频率:", self.freq_layout)
        
        # 条纹强度
        self.intensity_layout = QHBoxLayout()
        self.intensity_input = QSpinBox()
        self.intensity_input.setRange(1, 127)
        self.intensity_input.setValue(100)
        self.intensity_input.setToolTip("条纹强度影响信噪比：\n- 高强度：提高信噪比但可能导致亮区饱和\n- 低强度：可能在暗区难以检测条纹\n- 推荐值：80-100，需根据扫描物体表面反光特性调整")
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(1, 127)
        self.intensity_slider.setValue(100)
        self.intensity_slider.setToolTip(self.intensity_input.toolTip())
        self.intensity_layout.addWidget(self.intensity_input)
        self.intensity_layout.addWidget(self.intensity_slider)
        params_layout.addRow("条纹强度:", self.intensity_layout)
        
        # 亮度偏移
        self.offset_layout = QHBoxLayout()
        self.offset_input = QSpinBox()
        self.offset_input.setRange(1, 255)
        self.offset_input.setValue(128)
        self.offset_input.setToolTip("亮度偏移影响整体曝光：\n- 推荐值：128（中值）确保正弦波完整显示\n- 较高值用于暗色物体\n- 较低值用于亮色或高反光物体\n- 条纹强度+亮度偏移不应超过255")
        self.offset_slider = QSlider(Qt.Horizontal)
        self.offset_slider.setRange(1, 255)
        self.offset_slider.setValue(128)
        self.offset_slider.setToolTip(self.offset_input.toolTip())
        self.offset_layout.addWidget(self.offset_input)
        self.offset_layout.addWidget(self.offset_slider)
        params_layout.addRow("亮度偏移:", self.offset_layout)
        
        # 噪声水平
        self.noise_layout = QHBoxLayout()
        self.noise_input = QSpinBox()
        self.noise_input.setRange(0, 50)
        self.noise_input.setValue(0)
        self.noise_input.setToolTip("噪声水平用于模拟真实环境：\n- 实际应用中通常设置为0\n- 仅用于测试相位解包裹算法的抗噪性能\n- 值越高，图案越模糊，重建越困难")
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setRange(0, 50)
        self.noise_slider.setValue(0)
        self.noise_slider.setToolTip(self.noise_input.toolTip())
        self.noise_layout.addWidget(self.noise_input)
        self.noise_layout.addWidget(self.noise_slider)
        params_layout.addRow("噪声水平:", self.noise_layout)
        
        # 3. 保存设置组
        save_group = QGroupBox("保存设置")
        save_layout = QHBoxLayout(save_group)
        
        # 保存目录
        self.save_dir_input = QLineEdit("fringe_patterns")
        self.browse_button = QPushButton("浏览...")
        self.save_dir_input.setToolTip("指定保存生成图像的文件夹路径。")
        self.browse_button.setToolTip("点击选择一个文件夹来保存生成的条纹图像。")
        save_layout.addWidget(self.save_dir_input, 3)
        save_layout.addWidget(self.browse_button, 1)
        
        # 4. 操作按钮
        actions_layout = QHBoxLayout()
        
        # 帮助按钮
        self.help_button = QPushButton("参数指南")
        self.help_button.setStyleSheet("padding: 8px;")
        
        # 刷新预览按钮
        self.refresh_button = QPushButton("刷新预览")
        self.refresh_button.setStyleSheet("padding: 8px;")
        self.refresh_button.setToolTip("根据当前参数，更新右侧的预览图像。")
        
        # 生成图案按钮
        self.generate_button = QPushButton("生成图案")
        self.generate_button.setStyleSheet("background-color: #4a90e2; color: white; font-weight: bold; padding: 10px;")
        self.generate_button.setToolTip("根据当前设置，开始生成并保存相移条纹图像。")
        
        actions_layout.addWidget(self.help_button)
        actions_layout.addWidget(self.refresh_button)
        actions_layout.addWidget(self.generate_button)
        
        # 添加各组件到左侧设置面板
        settings_layout.addWidget(mode_group)
        settings_layout.addWidget(self.direction_group_box)
        settings_layout.addWidget(params_group)

        # 3D对象模拟设置组
        self.simulation_group = QGroupBox("3D对象模拟")
        self.simulation_group.setCheckable(True)
        self.simulation_group.setChecked(False)
        simulation_layout = QFormLayout(self.simulation_group)
        simulation_layout.setSpacing(10)
        
        self.simulation_group.setToolTip("启用后，将在生成的条纹上模拟一个3D对象，使其产生变形效果。")

        # 形状选择
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["球体", "锥体", "矩形", "多峰高斯"])
        self.shape_combo.setToolTip("选择要模拟的3D对象形状。")
        simulation_layout.addRow("对象形状:", self.shape_combo)

        # 调制强度
        self.modulation_layout = QHBoxLayout()
        self.modulation_input = QSpinBox()
        self.modulation_input.setRange(1, 500)
        self.modulation_input.setValue(30)
        self.modulation_input.setToolTip("3D对象对相位的调制强度。\n值越大，物体起伏越明显。范围 1-500。")
        self.modulation_slider = QSlider(Qt.Horizontal)
        self.modulation_slider.setRange(1, 500)
        self.modulation_slider.setValue(30)
        self.modulation_slider.setToolTip(self.modulation_input.toolTip())
        self.modulation_layout.addWidget(self.modulation_input)
        self.modulation_layout.addWidget(self.modulation_slider)
        simulation_layout.addRow("调制强度:", self.modulation_layout)
        
        settings_layout.addWidget(self.simulation_group)
        settings_layout.addWidget(save_group)
        settings_layout.addLayout(actions_layout)
        settings_layout.addStretch()
        
        # 右侧预览面板
        preview_title = QLabel("条纹预览")
        preview_title.setAlignment(Qt.AlignCenter)
        preview_title.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        # 预览图像区域
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #ffffff; border: 1px solid #e0e0e0;")
        self.preview_label.setMinimumSize(400, 300)
        
        # 预览信息
        self.info_label = QLabel("条纹方向: 垂直 | 频率: 15 | 相位步数: 1/4 (0°)")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("background-color: #f5f5f7; padding: 8px; border-radius: 4px;")
        
        # 相位选择
        phase_layout = QHBoxLayout()
        phase_layout.setSpacing(10)
        
        self.phase_label = QLabel("预览步数:")
        self.phase_input = QSpinBox()
        self.phase_input.setRange(1, 4) # 初始范围，会动态更新
        self.phase_slider = QSlider(Qt.Horizontal)
        self.phase_slider.setRange(1, 4) # 初始范围，会动态更新
        self.phase_input.setToolTip("选择预览第几步的相移图像。")
        self.phase_slider.setToolTip("拖动滑块以快速预览不同相移步骤的图像。")
        
        phase_layout.addWidget(self.phase_label)
        phase_layout.addWidget(self.phase_input)
        phase_layout.addWidget(self.phase_slider)
        
        # 状态提示
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #4a90e2; margin-top: 10px;")
        
        # 添加各组件到右侧预览面板
        preview_layout.addWidget(preview_title)
        preview_layout.addWidget(self.preview_label, 1)
        preview_layout.addLayout(phase_layout)
        preview_layout.addWidget(self.info_label)
        preview_layout.addWidget(self.status_label)
        
        # 生成初始预览
        self.update_preview()
    
    def connect_signals(self):
        """连接信号和槽"""
        # 模式切换
        self.single_mode_radio.toggled.connect(self.on_mode_change)
        
        # 条纹方向变化
        self.vertical_radio.toggled.connect(self.update_preview)
        self.horizontal_radio.toggled.connect(self.update_preview)
        
        # 参数变化
        self.steps_input.valueChanged.connect(self.update_phase_controls)
        self.width_input.valueChanged.connect(self.update_preview)
        self.height_input.valueChanged.connect(self.update_preview)
        self.freq_input.valueChanged.connect(self.sync_freq_slider)
        self.freq_slider.valueChanged.connect(self.sync_freq_input)
        self.intensity_input.valueChanged.connect(self.sync_intensity_slider)
        self.intensity_slider.valueChanged.connect(self.sync_intensity_input)
        self.offset_input.valueChanged.connect(self.sync_offset_slider)
        self.offset_slider.valueChanged.connect(self.sync_offset_input)
        self.noise_input.valueChanged.connect(self.sync_noise_slider)
        self.noise_slider.valueChanged.connect(self.sync_noise_input)
        
        # 相位变化
        self.phase_input.valueChanged.connect(self.sync_phase_slider_from_input)
        self.phase_slider.valueChanged.connect(self.sync_phase_input_from_slider)
        
        # 按钮点击
        self.help_button.clicked.connect(self.show_help)
        self.browse_button.clicked.connect(self.browse_directory)
        self.refresh_button.clicked.connect(self.update_preview)
        self.generate_button.clicked.connect(self.generate_patterns)
        
        # 3D模拟参数变化
        self.simulation_group.toggled.connect(self.update_preview)
        self.shape_combo.currentTextChanged.connect(self.update_preview)
        self.modulation_input.valueChanged.connect(self.sync_modulation_slider)
        self.modulation_slider.valueChanged.connect(self.sync_modulation_input)

    def on_mode_change(self, checked):
        """切换生成模式"""
        if checked: # single_mode_radio is checked
            self.direction_group_box.setEnabled(True)
            self.generate_button.setText("生成图案")
        else: # batch_mode_radio is checked
            self.direction_group_box.setEnabled(False)
            self.generate_button.setText("一键生成所有图案")
    
    def update_phase_controls(self, steps):
        """根据总步数N更新相位预览控件的范围"""
        # 保存当前值，如果可能的话
        current_step = self.phase_input.value()
        
        self.phase_input.setRange(1, steps)
        self.phase_slider.setRange(1, steps)

        # 恢复旧值，如果它在新的范围内
        if current_step <= steps:
            self.phase_input.setValue(current_step)
        
        self.update_preview()

    def sync_phase_slider_from_input(self, value):
        """从输入框同步相位滑块，避免循环"""
        self.phase_slider.blockSignals(True)
        self.phase_slider.setValue(value)
        self.phase_slider.blockSignals(False)
        self.update_preview()

    def sync_phase_input_from_slider(self, value):
        """从滑块同步相位输入框，避免循环"""
        self.phase_input.blockSignals(True)
        self.phase_input.setValue(value)
        self.phase_input.blockSignals(False)
        self.update_preview()

    def sync_modulation_slider(self, value):
        """同步调制强度滑块"""
        self.modulation_slider.setValue(value)
        self.update_preview()

    def sync_modulation_input(self, value):
        """同步调制强度输入框"""
        self.modulation_input.setValue(value)
        self.update_preview()
        
    def sync_freq_slider(self, value):
        """同步频率滑块"""
        self.freq_slider.setValue(value)
        self.update_preview()
    
    def sync_freq_input(self, value):
        """同步频率输入框"""
        self.freq_input.setValue(value)
        self.update_preview()
    
    def sync_intensity_slider(self, value):
        """同步强度滑块"""
        self.intensity_slider.setValue(value)
        self.update_preview()
    
    def sync_intensity_input(self, value):
        """同步强度输入框"""
        self.intensity_input.setValue(value)
        self.update_preview()
    
    def sync_offset_slider(self, value):
        """同步偏移滑块"""
        self.offset_slider.setValue(value)
        self.update_preview()
    
    def sync_offset_input(self, value):
        """同步偏移输入框"""
        self.offset_input.setValue(value)
        self.update_preview()
    
    def sync_noise_slider(self, value):
        """同步噪声滑块"""
        self.noise_slider.setValue(value)
        self.update_preview()
    
    def sync_noise_input(self, value):
        """同步噪声输入框"""
        self.noise_input.setValue(value)
        self.update_preview()
    
    def browse_directory(self):
        """浏览保存目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if directory:
            self.save_dir_input.setText(directory)

    def _create_3d_object_height_map(self, width, height):
        """根据UI设置创建3D对象的高度图 (zz)"""
        if not self.simulation_group.isChecked():
            return np.zeros((height, width))

        shape = self.shape_combo.currentText()
        modulation = self.modulation_input.value() / 10.0

        cx, cy = width // 2, height // 2
        zz = np.zeros((height, width))
        
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)

        if shape == "球体":
            r = min(width, height) // 3
            d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            mask = d < r
            if np.any(mask):
                zz[mask] = np.sqrt(r**2 - d[mask]**2)
            
        elif shape == "锥体":
            r = min(width, height) // 3
            d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            mask = d < r
            if np.any(mask):
                zz[mask] = r - d[mask]

        elif shape == "矩形":
            rect_width = width // 3
            rect_height = height // 3
            x1, x2 = cx - rect_width // 2, cx + rect_width // 2
            y1, y2 = cy - rect_height // 2, cy + rect_height // 2
            zz[int(y1):int(y2), int(x1):int(x2)] = min(width, height) // 4
            
        elif shape == "多峰高斯":
            peaks = [
                (cx - width//4, cy - height//4, width//8, height//8, 1),
                (cx + width//4, cy + height//4, width//9, height//9, 0.8),
                (cx + width//5, cy - height//5, width//12, height//12, 0.9)
            ]
            for (peak_cx, peak_cy, sig_x, sig_y, h) in peaks:
                gauss = h * np.exp(-(((xx - peak_cx)**2 / (2 * sig_x**2)) + ((yy - peak_cy)**2 / (2 * sig_y**2))))
                zz += gauss * min(width, height) / 2

        if np.max(zz) > 0:
            zz = (zz / np.max(zz)) * modulation
        
        return zz

    def update_preview(self):
        """更新预览图像"""
        # 获取当前参数
        direction = "horizontal" if self.horizontal_radio.isChecked() else "vertical"
        steps = self.steps_input.value()
        frequency = self.freq_input.value()
        intensity = self.intensity_input.value()
        offset = self.offset_input.value()
        noise_level = self.noise_input.value()
        
        # 确定预览相位
        phase_step = self.phase_input.value()
        phase_shift = (phase_step - 1) * 2 * math.pi / steps
        
        # 更新信息标签
        phase_degree = int(phase_shift * 180 / math.pi)
        self.info_label.setText(f"条纹方向: {'水平' if direction == 'horizontal' else '垂直'} | 频率: {frequency} | 相位: {phase_step}/{steps} ({phase_degree}°)")
        
        # 生成预览图像
        preview_width, preview_height = 400, 300
        x_coords, y_coords = np.meshgrid(np.linspace(0, 1, preview_width), np.linspace(0, 1, preview_height))
        
        # 生成3D对象高度图
        zz = self._create_3d_object_height_map(preview_width, preview_height)

        if direction == "horizontal":
            # 水平条纹
            fringe = offset + intensity * np.sin(2 * math.pi * frequency * y_coords + phase_shift + zz)
        else:
            # 垂直条纹
            fringe = offset + intensity * np.sin(2 * math.pi * frequency * x_coords + phase_shift + zz)
        
        # 添加噪声
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, fringe.shape)
            fringe = fringe + noise
        
        # 限制范围
        fringe = np.clip(fringe, 0, 255).astype(np.uint8)
        
        # 转换为QImage
        qimg = QImage(fringe.data, preview_width, preview_height, preview_width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        
        # 显示预览图像
        self.preview_label.setPixmap(pixmap)
    
    def generate_patterns(self):
        """生成N步相移图案"""
        try:
            # 获取参数
            steps = self.steps_input.value()
            width = self.width_input.value()
            height = self.height_input.value()
            frequency = self.freq_input.value()
            intensity = self.intensity_input.value()
            offset = self.offset_input.value()
            noise_level = self.noise_input.value()
            save_dir = self.save_dir_input.text()

            # 确定要处理的方向
            directions_to_process = []
            if self.batch_mode_radio.isChecked():
                directions_to_process = ["vertical", "horizontal"]
            else:
                direction = "horizontal" if self.horizontal_radio.isChecked() else "vertical"
                directions_to_process.append(direction)
            
            # 创建保存目录
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # 生成网格
            x_coords, y_coords = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))

            # 生成3D对象高度图
            zz = self._create_3d_object_height_map(width, height)

            # 生成图像
            total_images_generated = 0
            for direction in directions_to_process:
                self.status_label.setText(f"正在生成 {direction} 条纹...")
                QApplication.processEvents()

                for i in range(steps):
                    # 计算相移量
                    phase_shift = i * 2 * math.pi / steps
                    
                    # 根据方向生成条纹和文件名
                    if direction == "horizontal":
                        fringe = offset + intensity * np.sin(2 * math.pi * frequency * y_coords + phase_shift + zz)
                        filename_prefix = f"I{i + 1 + steps}"
                    else: # vertical
                        fringe = offset + intensity * np.sin(2 * math.pi * frequency * x_coords + phase_shift + zz)
                        filename_prefix = f"I{i + 1}"
                    
                    # 添加噪声
                    if noise_level > 0:
                        noise = np.random.normal(0, noise_level, fringe.shape)
                        fringe += noise
                    
                    # 确保像素值在[0, 255]范围内
                    fringe = np.clip(fringe, 0, 255).astype(np.uint8)
                    
                    # 保存图像
                    filename = os.path.join(save_dir, f"{filename_prefix}.png")
                    cv.imwrite(filename, fringe)
                    
                    phase_degrees = int(phase_shift * 180 / math.pi)
                    self.status_label.setText(f"已生成 {direction} 条纹: {filename_prefix}.png (第{i+1}/{steps}步)")
                    QApplication.processEvents()
                total_images_generated += steps

            # 显示成功消息
            self.status_label.setText(f"所有图案生成完成！已保存到 {save_dir} 目录")
            
            # 显示完成对话框
            msg_box = QMessageBox()
            msg_box.setWindowTitle("生成完成")
            
            if len(directions_to_process) > 1:
                pattern_desc = f"垂直条纹 (I1-I{steps}) 和水平条纹 (I{steps+1}-I{2*steps}) 均已生成。"
            else:
                direction = directions_to_process[0]
                if direction == "horizontal":
                    pattern_desc = f"水平条纹图案 (I{steps+1}-I{2*steps}) 已生成。"
                else:
                    pattern_desc = f"垂直条紋图案 (I1-I{steps}) 已生成。"
                
            msg_box.setText(f"{pattern_desc}\n\n"
                           f"图像已保存到: {save_dir}\n"
                           f"图像尺寸: {width}x{height}像素\n"
                           f"条纹频率: {frequency}")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec()
            
        except Exception as e:
            # 显示错误消息
            self.status_label.setText(f"生成失败: {str(e)}")
            
            msg_box = QMessageBox()
            msg_box.setWindowTitle("错误")
            msg_box.setText(f"生成图案时出错:\n\n{str(e)}")
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.exec()

    def show_help(self):
        """显示参数指南帮助对话框"""
        help_msg = QMessageBox(self)
        help_msg.setWindowTitle("结构光参数指南")
        help_msg.setIcon(QMessageBox.Information)
        
        help_text = """<h3>N步相移条纹参数指南</h3>
        
<p><b>相移步数 (N)</b>：一个周期内相移的次数
<ul>
<li><b>影响</b>：步数越多，相位计算越精确，抗噪能力越强，但需要采集和处理的图像也越多。</li>
<li><b>推荐值</b>：通常选择3, 4或8步。4步相移是最经典和常用的方法。</li>
</ul></p>

<p><b>条纹频率</b>：决定一个完整周期内的条纹数量
<ul>
<li><b>影响</b>：直接影响相位分辨率和3D重建精度</li>
<li><b>高频率(20-30)</b>：提供更高的空间分辨率和精度，但增加相位解包裹错误的风险</li>
<li><b>低频率(8-15)</b>：相位解包裹更稳健，但空间分辨率较低</li>
<li><b>推荐策略</b>：先用低频率进行整体扫描，再用高频率进行精细区域扫描</li>
</ul></p>

<p><b>条纹强度</b>：控制正弦条纹的振幅
<ul>
<li><b>影响</b>：决定图案的对比度和信噪比</li>
<li><b>高强度</b>：提高信噪比，适用于低反光表面，但可能导致亮区饱和</li>
<li><b>低强度</b>：避免饱和，但可能在暗区难以检测条纹</li>
<li><b>推荐值</b>：80-100，需要根据被扫描物体表面的反光特性调整</li>
</ul></p>

<p><b>亮度偏移</b>：控制整体基线亮度
<ul>
<li><b>影响</b>：决定整体曝光水平，确保条纹在整个物体表面可见</li>
<li><b>标准值</b>：128（中间值）确保正弦波完整显示</li>
<li><b>调整原则</b>：条纹强度 + 亮度偏移应不超过255，以避免过曝</li>
<li><b>场景调整</b>：暗色物体使用较高偏移，亮色或高反光物体使用较低偏移</li>
</ul></p>

<p><b>最佳实践</b>：
<ol>
<li>对于未知物体，先使用默认参数进行测试扫描</li>
<li>根据扫描结果调整参数：对于过暗区域增加偏移，对于过亮区域降低偏移</li>
<li>如果出现条纹无法识别，适当增加条纹强度</li>
<li>为获得最佳结果，可能需要对不同反光特性的区域进行多次扫描</li>
</ol></p>
"""
        help_msg.setText(help_text)
        help_msg.setTextFormat(Qt.RichText)
        help_msg.exec()


def main():
    app = QApplication(sys.argv)
    window = FringePatternGenerator()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 