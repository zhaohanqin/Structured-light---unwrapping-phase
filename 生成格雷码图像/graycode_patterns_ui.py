import sys
import os
import numpy as np
import cv2 as cv
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QRadioButton, 
                             QPushButton, QSpinBox, QGroupBox, QButtonGroup, 
                             QSlider, QFileDialog, QMessageBox, QFormLayout)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QSize
from PySide6.QtGui import QPixmap, QImage, QColor, QPalette, QFont

# 导入GrayCode类
from generate_graycode_patterns import GrayCode

class GrayCodePatternGenerator(QMainWindow):
    """格雷码条纹图案生成器的UI界面"""
    
    def __init__(self):
        super().__init__()
        
        # 设置窗口标题和大小
        self.setWindowTitle("格雷码条纹图案生成器")
        self.resize(900, 650)
        
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
        self.single_mode_radio.setToolTip("只生成指定方向（垂直或水平）的格雷码图案。")
        self.batch_mode_radio.setToolTip("一次性生成垂直和水平两个方向的格雷码图案，\n并分别保存到对应的子目录中。")
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
        self.vertical_radio = QRadioButton("垂直条纹（用于水平方向解包裹）")
        self.horizontal_radio = QRadioButton("水平条纹（用于垂直方向解包裹）")
        self.vertical_radio.setToolTip("生成的条纹是垂直的，用于解码沿水平方向变化的相位。")
        self.horizontal_radio.setToolTip("生成的条纹是水平的，用于解码沿垂直方向变化的相位。")
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
        params_info = QLabel("调整这些参数可以优化格雷码图案。点击\"参数指南\"获取详细建议。")
        params_info.setStyleSheet("color: #555; font-style: italic; padding: 4px;")
        params_info.setWordWrap(True)
        params_layout.addRow(params_info)
        
        # 图像宽度
        self.width_input = QSpinBox()
        self.width_input.setRange(100, 4096)
        self.width_input.setValue(1024)
        self.width_input.setSingleStep(10)
        self.width_input.setToolTip("设置生成图像的宽度（像素）。\n建议与投影仪或相机的分辨率保持一致。")
        params_layout.addRow("图像宽度:", self.width_input)
        
        # 图像高度
        self.height_input = QSpinBox()
        self.height_input.setRange(100, 4096)
        self.height_input.setValue(768)
        self.height_input.setSingleStep(10)
        self.height_input.setToolTip("设置生成图像的高度（像素）。\n建议与投影仪或相机的分辨率保持一致。")
        params_layout.addRow("图像高度:", self.height_input)
        
        # 格雷码位数
        self.bits_layout = QHBoxLayout()
        self.bits_input = QSpinBox()
        self.bits_input.setRange(1, 10)
        self.bits_input.setValue(5)
        self.bits_input.setToolTip("格雷码位数影响条纹密度：\n- 位数越高，条纹越密集，分辨率越高\n- 位数越低，条纹越稀疏，抗干扰能力越强\n- 推荐值：5位（常用）")
        self.bits_slider = QSlider(Qt.Horizontal)
        self.bits_slider.setRange(1, 10)
        self.bits_slider.setValue(5)
        self.bits_slider.setToolTip(self.bits_input.toolTip())
        self.bits_layout.addWidget(self.bits_input)
        self.bits_layout.addWidget(self.bits_slider)
        params_layout.addRow("格雷码位数:", self.bits_layout)
        
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
        save_layout = QVBoxLayout(save_group)
        
        # 保存目录说明
        save_info = QLabel("根据条纹方向，图案将保存到以下目录：")
        save_info.setWordWrap(True)
        save_layout.addWidget(save_info)
        
        # 保存目录信息
        self.save_dir_label = QLabel("垂直条纹 → gray_patterns/gray_patterns_horizontal\n水平条纹 → gray_patterns/gray_patterns_vertical")
        self.save_dir_label.setStyleSheet("background-color: #f5f5f7; padding: 8px; border-radius: 4px;")
        save_layout.addWidget(self.save_dir_label)
        
        # 自定义保存目录
        custom_dir_layout = QHBoxLayout()
        self.custom_dir_check = QRadioButton("自定义保存目录:")
        self.save_dir_input = QLineEdit("")
        self.save_dir_input.setEnabled(False)
        self.save_dir_input.setToolTip("指定一个自定义的根目录来保存生成的格雷码文件夹。")
        self.browse_button = QPushButton("浏览...")
        self.browse_button.setEnabled(False)
        self.browse_button.setToolTip("点击选择一个自定义的保存位置。")
        custom_dir_layout.addWidget(self.custom_dir_check)
        custom_dir_layout.addWidget(self.save_dir_input, 3)
        custom_dir_layout.addWidget(self.browse_button, 1)
        save_layout.addLayout(custom_dir_layout)
        
        # 默认目录选项
        self.default_dir_check = QRadioButton("使用默认目录")
        self.default_dir_check.setChecked(True)
        self.default_dir_check.setToolTip("程序将在当前目录下自动创建 'gray_patterns' 文件夹来保存结果。")
        save_layout.addWidget(self.default_dir_check)
        
        # 4. 操作按钮
        actions_layout = QHBoxLayout()
        
        # 帮助按钮
        self.help_button = QPushButton("参数指南")
        self.help_button.setStyleSheet("padding: 8px;")
        
        # 刷新预览按钮
        self.refresh_button = QPushButton("刷新预览")
        self.refresh_button.setStyleSheet("padding: 8px;")
        self.refresh_button.setToolTip("根据当前设置的参数，更新右侧的预览图像。")
        
        # 生成图案按钮
        self.generate_button = QPushButton("生成图案")
        self.generate_button.setStyleSheet("background-color: #4a90e2; color: white; font-weight: bold; padding: 10px;")
        self.generate_button.setToolTip("根据当前设置，开始生成并保存格雷码图像文件。")
        
        actions_layout.addWidget(self.help_button)
        actions_layout.addWidget(self.refresh_button)
        actions_layout.addWidget(self.generate_button)
        
        # 添加各组件到左侧设置面板
        settings_layout.addWidget(mode_group)
        settings_layout.addWidget(self.direction_group_box)
        settings_layout.addWidget(params_group)
        settings_layout.addWidget(save_group)
        settings_layout.addLayout(actions_layout)
        settings_layout.addStretch()
        
        # 右侧预览面板
        preview_title = QLabel("格雷码预览")
        preview_title.setAlignment(Qt.AlignCenter)
        preview_title.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        # 预览图像区域
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #ffffff; border: 1px solid #e0e0e0;")
        self.preview_label.setMinimumSize(400, 300)
        
        # 预览信息
        self.info_label = QLabel("条纹方向: 垂直 | 格雷码位数: 5 | 位索引: 0")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("background-color: #f5f5f7; padding: 8px; border-radius: 4px;")
        
        # 位索引选择
        bit_layout = QHBoxLayout()
        bit_layout.setSpacing(10)
        
        self.bit_label = QLabel("预览位索引:")
        self.bit_slider = QSlider(Qt.Horizontal)
        self.bit_slider.setRange(0, 4)  # 默认5位，索引0-4
        self.bit_slider.setValue(0)
        self.bit_slider.setTickPosition(QSlider.TicksBelow)
        self.bit_slider.setTickInterval(1)
        self.bit_slider.setToolTip("拖动滑块以预览不同位数的格雷码图案。")
        
        self.bit_index_label = QLabel("0")
        self.bit_index_label.setAlignment(Qt.AlignCenter)
        self.bit_index_label.setMinimumWidth(20)
        
        bit_layout.addWidget(self.bit_label)
        bit_layout.addWidget(self.bit_slider)
        bit_layout.addWidget(self.bit_index_label)
        
        # 状态提示
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #4a90e2; margin-top: 10px;")
        
        # 添加各组件到右侧预览面板
        preview_layout.addWidget(preview_title)
        preview_layout.addWidget(self.preview_label, 1)
        preview_layout.addLayout(bit_layout)
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
        self.horizontal_radio.toggled.connect(self.update_direction_info)
        
        # 参数变化
        self.bits_input.valueChanged.connect(self.sync_bits_slider)
        self.bits_slider.valueChanged.connect(self.sync_bits_input)
        self.noise_input.valueChanged.connect(self.sync_noise_slider)
        self.noise_slider.valueChanged.connect(self.sync_noise_input)
        
        # 位索引变化
        self.bit_slider.valueChanged.connect(self.update_bit_index)
        
        # 保存目录选项
        self.custom_dir_check.toggled.connect(self.toggle_custom_dir)
        self.default_dir_check.toggled.connect(self.toggle_default_dir)
        
        # 按钮点击
        self.help_button.clicked.connect(self.show_help)
        self.browse_button.clicked.connect(self.browse_directory)
        self.refresh_button.clicked.connect(self.update_preview)
        self.generate_button.clicked.connect(self.generate_patterns)
    
    def on_mode_change(self, checked):
        """切换生成模式"""
        if checked: # single_mode_radio is checked
            self.direction_group_box.setEnabled(True)
            self.generate_button.setText("生成图案")
        else: # batch_mode_radio is checked
            self.direction_group_box.setEnabled(False)
            self.generate_button.setText("一键生成所有图案")
    
    def sync_bits_slider(self, value):
        """同步位数滑块"""
        self.bits_slider.setValue(value)
        # 更新位索引滑块的范围
        self.bit_slider.setRange(0, value - 1)
        # 如果当前位索引超出范围，重置为0
        if self.bit_slider.value() >= value:
            self.bit_slider.setValue(0)
        self.update_preview()
    
    def sync_bits_input(self, value):
        """同步位数输入框"""
        self.bits_input.setValue(value)
        # 更新位索引滑块的范围
        self.bit_slider.setRange(0, value - 1)
        # 如果当前位索引超出范围，重置为0
        if self.bit_slider.value() >= value:
            self.bit_slider.setValue(0)
        self.update_preview()
    
    def sync_noise_slider(self, value):
        """同步噪声滑块"""
        self.noise_slider.setValue(value)
        self.update_preview()
    
    def sync_noise_input(self, value):
        """同步噪声输入框"""
        self.noise_input.setValue(value)
        self.update_preview()
    
    def update_bit_index(self, value):
        """更新位索引"""
        self.bit_index_label.setText(str(value))
        self.update_preview()
    
    def update_direction_info(self):
        """更新方向信息"""
        if self.horizontal_radio.isChecked():
            self.save_dir_label.setText("水平条纹 → gray_patterns/gray_patterns_vertical")
        else:
            self.save_dir_label.setText("垂直条纹 → gray_patterns/gray_patterns_horizontal\n垂直条纹 → gray_patterns (兼容性考虑)")
        self.update_preview()
    
    def toggle_custom_dir(self, checked):
        """切换自定义目录选项"""
        self.save_dir_input.setEnabled(checked)
        self.browse_button.setEnabled(checked)
        if checked:
            self.default_dir_check.setChecked(False)
    
    def toggle_default_dir(self, checked):
        """切换默认目录选项"""
        if checked:
            self.custom_dir_check.setChecked(False)
            self.save_dir_input.setEnabled(False)
            self.browse_button.setEnabled(False)
    
    def browse_directory(self):
        """浏览保存目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if directory:
            self.save_dir_input.setText(directory)
    
    def update_preview(self):
        """更新预览图像"""
        # 获取当前参数
        direction = "horizontal" if self.horizontal_radio.isChecked() else "vertical"
        bits = self.bits_input.value()
        bit_index = self.bit_slider.value()
        noise_level = self.noise_input.value()
        
        # 更新信息标签
        self.info_label.setText(f"条纹方向: {'水平' if direction == 'horizontal' else '垂直'} | 格雷码位数: {bits} | 位索引: {bit_index}")
        
        # 创建格雷码生成器
        g = GrayCode(bits)
        
        # 生成预览图像
        preview_width, preview_height = 400, 300
        
        try:
            if direction == "horizontal":
                # 水平条纹
                pattern = g.toHorizontalPattern(bit_index, preview_width, preview_height)
            else:
                # 垂直条纹
                pattern = g.toPattern(bit_index, preview_width, preview_height)
            
            # 添加噪声
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, pattern.shape)
                pattern = pattern + noise
                pattern = np.clip(pattern, 0, 255).astype(np.uint8)
            
            # 转换为QImage
            qimg = QImage(pattern.data, pattern.shape[1], pattern.shape[0], pattern.shape[1], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg)
            
            # 显示预览图像
            self.preview_label.setPixmap(pixmap)
            self.status_label.setText("预览已更新")
            
        except Exception as e:
            self.status_label.setText(f"预览生成失败: {str(e)}")
            print(f"预览生成错误: {e}")
    
    def generate_patterns(self):
        """生成格雷码条纹图案"""
        try:
            # 获取参数
            width = self.width_input.value()
            height = self.height_input.value()
            bits = self.bits_input.value()
            noise_level = self.noise_input.value()
            
            directions_to_process = []
            if self.batch_mode_radio.isChecked():
                directions_to_process = ["vertical", "horizontal"]
            else:
                direction = "horizontal" if self.horizontal_radio.isChecked() else "vertical"
                directions_to_process.append(direction)

            saved_dirs_info = []

            for direction in directions_to_process:
                # 确定保存目录
                if self.custom_dir_check.isChecked() and self.save_dir_input.text():
                    # 在批处理模式下，为每个方向创建子目录
                    if self.batch_mode_radio.isChecked():
                        base_dir = self.save_dir_input.text()
                        sub_dir = "gray_patterns_vertical" if direction == "horizontal" else "gray_patterns_horizontal"
                        save_dir = os.path.join(base_dir, sub_dir)
                    else:
                        save_dir = self.save_dir_input.text()
                else:
                    save_dir = None  # 使用默认目录
                
                # 创建格雷码生成器
                g = GrayCode(bits)
                
                # 确定实际保存目录（用于显示）
                if save_dir is None:
                    if direction == "horizontal":
                        actual_dir = "gray_patterns/gray_patterns_vertical"
                    else:
                        actual_dir = "gray_patterns/gray_patterns_horizontal"
                else:
                    actual_dir = save_dir
                
                # 创建保存目录
                os.makedirs(actual_dir, exist_ok=True)
                
                # 如果是垂直条纹，还需要创建gray_patterns目录
                if direction == "vertical" and (save_dir is None or "gray_patterns" not in save_dir):
                    os.makedirs("gray_patterns", exist_ok=True)
                
                # 生成格雷码图像
                self.status_label.setText(f"正在生成 {direction} 条纹...")
                QApplication.processEvents()
                
                for i in range(bits):
                    # 根据指定的方向生成条纹
                    if direction == "horizontal":
                        pattern = g.toHorizontalPattern(i, width, height)
                    else:
                        pattern = g.toPattern(i, width, height)
                    
                    # 添加噪声（如果指定）
                    if noise_level > 0:
                        noise = np.random.normal(0, noise_level, pattern.shape)
                        pattern = np.clip(pattern + noise, 0, 255).astype(np.uint8)
                    
                    # 添加轻微高斯模糊（模拟光学系统）
                    pattern_blurred = cv.GaussianBlur(pattern, (3, 3), 0.5)
                    
                    # 生成二值化图像
                    _, binary_pattern = cv.threshold(pattern_blurred, 127, 255, cv.THRESH_BINARY)
                    
                    # 保存原始格雷码图案
                    gray_filename = os.path.join(actual_dir, f"gray_bit_{i}.png")
                    cv.imwrite(gray_filename, pattern_blurred)
                    
                    # 保存二值化格雷码图案
                    binary_filename = os.path.join(actual_dir, f"matched_binary_{i}.png")
                    cv.imwrite(binary_filename, binary_pattern)
                    
                    # 如果是垂直条纹，复制到gray_patterns目录（兼容性考虑）
                    if direction == "vertical" and (save_dir is None or "gray_patterns" not in save_dir):
                        cv.imwrite(f"gray_patterns/gray_bit_{i}.png", pattern_blurred)
                        cv.imwrite(f"gray_patterns/matched_binary_{i}.png", binary_pattern)
                    
                    self.status_label.setText(f"已生成 {direction} 图像: (位 {i+1}/{bits})")
                    QApplication.processEvents()

                saved_dirs_info.append(actual_dir)
            
            # 显示成功消息
            self.status_label.setText(f"格雷码条纹图案生成完成！")
            
            # 显示完成对话框
            msg_box = QMessageBox()
            msg_box.setWindowTitle("生成完成")
            
            if len(saved_dirs_info) > 1:
                pattern_desc = "垂直和水平条纹格雷码图案均已生成。"
                dirs_info = "图像已分别保存到以下目录:\n- {}\n- {}".format(*saved_dirs_info)
            else:
                direction = directions_to_process[0]
                if direction == "horizontal":
                    pattern_desc = "水平条纹格雷码图案（用于垂直方向解包裹）"
                else:
                    pattern_desc = "垂直条纹格雷码图案（用于水平方向解包裹）"
                dirs_info = f"图像已保存到: {saved_dirs_info[0]}"

            msg_box.setText(f"{pattern_desc}\n\n"
                           f"{dirs_info}\n\n"
                           f"图像尺寸: {width}x{height}像素\n"
                           f"格雷码位数: {bits}")
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
        help_msg.setWindowTitle("格雷码参数指南")
        help_msg.setIcon(QMessageBox.Information)
        
        help_text = """<h3>格雷码条纹参数指南</h3>
        
<p><b>条纹方向</b>：决定格雷码条纹的延伸方向
<ul>
<li><b>垂直条纹</b>：条纹沿垂直方向延伸，用于水平方向解包裹</li>
<li><b>水平条纹</b>：条纹沿水平方向延伸，用于垂直方向解包裹</li>
<li><b>选择原则</b>：根据您需要解包裹的方向选择相应的条纹方向</li>
</ul></p>

<p><b>格雷码位数</b>：决定条纹的精细程度
<ul>
<li><b>影响</b>：位数越高，条纹越密集，分辨率越高</li>
<li><b>高位数(7-10)</b>：提供更高的空间分辨率，但可能增加解码错误的风险</li>
<li><b>低位数(3-4)</b>：更稳健，但空间分辨率较低</li>
<li><b>推荐值</b>：5位（常用标准，平衡精度和稳健性）</li>
</ul></p>

<p><b>噪声水平</b>：模拟真实环境中的噪声
<ul>
<li><b>影响</b>：增加噪声可以测试算法的抗干扰能力</li>
<li><b>推荐值</b>：实际应用中通常设置为0</li>
<li><b>测试用途</b>：仅用于测试相位解包裹算法的抗噪性能</li>
</ul></p>

<p><b>保存目录</b>：
<ul>
<li><b>垂直条纹</b>：保存到 gray_patterns/gray_patterns_horizontal 和 gray_patterns</li>
<li><b>水平条纹</b>：保存到 gray_patterns/gray_patterns_vertical</li>
<li><b>目录命名说明</b>：目录名表示用于哪个方向的解包裹，而非条纹方向</li>
</ul></p>

<p><b>最佳实践</b>：
<ol>
<li>通常需要同时生成水平和垂直两个方向的格雷码图案</li>
<li>格雷码位数应与四步相移条纹的频率相匹配</li>
<li>确保投影设备分辨率足够显示最小的格雷码条纹</li>
<li>格雷码和四步相移图案应使用相同的图像分辨率</li>
</ol></p>
"""
        help_msg.setText(help_text)
        help_msg.setTextFormat(Qt.RichText)
        help_msg.exec()


def main():
    app = QApplication(sys.argv)
    window = GrayCodePatternGenerator()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 