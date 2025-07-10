import sys
import os
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QComboBox, QSpinBox, QLineEdit, 
                               QPushButton, QFileDialog, QRadioButton, QButtonGroup, 
                               QGroupBox, QTabWidget, QSplitter, QMessageBox, 
                               QProgressBar, QScrollArea, QSizePolicy)
from PySide6.QtGui import QPixmap, QImage, QColor, QPalette
from PySide6.QtCore import Qt, Signal, Slot, QThread, QDir
import cv2 as cv
from pathlib import Path
import glob

# 导入wrapped_phase模块
from wrapped_phase import WrappedPhase, normalize_image_size

class ImageProcessingThread(QThread):
    """用于后台处理图像的线程"""
    progress_update = Signal(int, str)
    processing_complete = Signal(object, object, str) # v_phase, h_phase, prefix_base
    processing_error = Signal(str)
    
    def __init__(self, wrapped_phase_instance, v_images, h_images):
        super().__init__()
        self.wp = wrapped_phase_instance
        self.v_images = v_images
        self.h_images = h_images
        
    def run(self):
        try:
            v_phase_result = None
            h_phase_result = None
            
            # 1. 处理垂直条纹
            if self.v_images and len(self.v_images) >= 3:
                self.progress_update.emit(10, "计算垂直条纹的包裹相位...")
                v_phase_result = self.wp.computeWrappedphase(self.v_images)
                self.progress_update.emit(50, "垂直条纹计算完成。")
            else:
                self.progress_update.emit(50, "无垂直条纹图像，跳过计算。")

            # 2. 处理水平条纹
            if self.h_images and len(self.h_images) >= 3:
                self.progress_update.emit(60, "计算水平条纹的包裹相位...")
                h_phase_result = self.wp.computeWrappedphase(self.h_images)
                self.progress_update.emit(90, "水平条纹计算完成。")
            else:
                self.progress_update.emit(90, "无水平条纹图像，跳过计算。")

            # 3. 发送完成信号
            self.processing_complete.emit(v_phase_result, h_phase_result, f"{self.wp.n}step_")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.processing_error.emit(f"处理图像时出错: {str(e)}")

class PhaseImageWidget(QWidget):
    """用于显示相位图像的组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.image_label = QLabel("暂无图像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ddd;")
        
        self.layout.addWidget(self.image_label)
        self.pixmap = None
        
    def set_image(self, cv_img):
        """设置要显示的OpenCV格式图像"""
        if cv_img is None:
            return
            
        h, w = cv_img.shape[:2]
        
        # 确保是三通道RGB图像
        if len(cv_img.shape) == 2:  # 灰度图
            cv_img = cv.cvtColor(cv_img, cv.COLOR_GRAY2RGB)
        elif cv_img.shape[2] == 3:  # BGR图像
            cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        
        # 创建QImage
        qimg = QImage(cv_img.data, w, h, cv_img.strides[0], QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg)
        
        # 设置缩放策略以适应小部件尺寸
        self.update_image_display()
        
    def update_image_display(self):
        """根据控件大小更新图像显示"""
        if self.pixmap is None:
            return
            
        # 缩放图像以适应控件大小，同时保持纵横比
        scaled_pixmap = self.pixmap.scaled(
            self.image_label.width(), 
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled_pixmap)
        
    def resizeEvent(self, event):
        """重写调整大小事件以更新图像显示"""
        super().resizeEvent(event)
        self.update_image_display()

class PreviewWidget(QWidget):
    """预览已加载图像的组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # 图像滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QHBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        
        self.layout.addWidget(QLabel("已加载图像预览:"))
        self.layout.addWidget(self.scroll_area)
        
        self.preview_labels = []
        
    def set_images(self, images):
        """设置要预览的图像列表"""
        # 清除现有的预览标签
        for label in self.preview_labels:
            self.scroll_layout.removeWidget(label)
            label.deleteLater()
        self.preview_labels = []
        
        if not images or len(images) == 0:
            empty_label = QLabel("暂无图像")
            empty_label.setAlignment(Qt.AlignCenter)
            self.scroll_layout.addWidget(empty_label)
            self.preview_labels.append(empty_label)
            return
            
        # 创建新的预览标签
        for i, img in enumerate(images):
            if img is None:
                continue
                
            # 创建预览标签
            preview_widget = QWidget()
            preview_layout = QVBoxLayout(preview_widget)
            
            # 图像标签
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setMinimumSize(120, 120)
            image_label.setMaximumSize(150, 150)
            
            # 转换为RGB并缩放
            if len(img.shape) == 2:  # 灰度图
                display_img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
            else:  # 彩色图
                display_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                
            # 缩放图像以适应预览
            h, w = display_img.shape[:2]
            aspect_ratio = w / h
            preview_w = min(120, w)
            preview_h = int(preview_w / aspect_ratio)
            if preview_h > 120:
                preview_h = 120
                preview_w = int(preview_h * aspect_ratio)
                
            display_img = cv.resize(display_img, (preview_w, preview_h))
            
            # 创建QImage和QPixmap
            qimg = QImage(display_img.data, preview_w, preview_h, 
                         display_img.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            image_label.setPixmap(pixmap)
            preview_layout.addWidget(image_label)
            
            # 图像索引标签
            index_label = QLabel(f"图像 {i+1}")
            index_label.setAlignment(Qt.AlignCenter)
            preview_layout.addWidget(index_label)
            
            self.scroll_layout.addWidget(preview_widget)
            self.preview_labels.append(preview_widget)

class WrappedPhaseUI(QMainWindow):
    """包裹相位计算的主UI窗口"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
        # 初始化包裹相位计算器
        self.wrapped_phase = WrappedPhase()
        self.loaded_images = []
        self.wrapped_phase_result = None
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("结构光包裹相位计算工具")
        self.setMinimumSize(1100, 600)  # 调整为更宽、更矮的窗口
        
        # 设置浅色系风格
        self.set_light_style()
        
        # 创建中央部件
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # 创建水平分割器用于左右布局
        main_splitter = QSplitter(Qt.Horizontal)
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        
        # 右侧结果显示面板
        result_panel = self.create_result_panel()
        
        # 添加到水平分割器
        main_splitter.addWidget(control_panel)
        main_splitter.addWidget(result_panel)
        main_splitter.setStretchFactor(0, 1)  # 控制面板占较少空间
        main_splitter.setStretchFactor(1, 2)  # 结果面板占更多空间
        
        main_layout.addWidget(main_splitter)
        
        # 底部状态栏
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage("就绪")
        
        # 设置中央部件
        self.setCentralWidget(central_widget)
        
    def set_light_style(self):
        """设置浅色系风格"""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 245))
        palette.setColor(QPalette.WindowText, QColor(70, 70, 70))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(245, 245, 250))
        palette.setColor(QPalette.Button, QColor(240, 240, 245))
        palette.setColor(QPalette.ButtonText, QColor(70, 70, 70))
        self.setPalette(palette)
        
        # 设置样式表
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #c0c0c0;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 3px;
            }
            
            QPushButton {
                background-color: #e6e6e6;
                border: 1px solid #c0c0c0;
                border-radius: 3px;
                padding: 5px 15px;
            }
            
            QPushButton:hover {
                background-color: #d9d9d9;
            }
            
            QPushButton:pressed {
                background-color: #cccccc;
            }
            
            QLineEdit, QComboBox, QSpinBox {
                border: 1px solid #c0c0c0;
                border-radius: 3px;
                padding: 3px;
                background-color: white;
            }
            
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                border-radius: 3px;
            }
            
            QTabBar::tab {
                background-color: #e6e6e6;
                border: 1px solid #c0c0c0;
                border-bottom-color: none;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                padding: 5px 10px;
            }
            
            QTabBar::tab:selected {
                background-color: white;
            }
            
            QTabBar::tab:!selected {
                margin-top: 2px;
            }
        """)
        
    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # 模式选择
        mode_group = QGroupBox("处理模式")
        mode_layout = QHBoxLayout(mode_group)
        self.mode_group = QButtonGroup(self)
        self.single_mode_radio = QRadioButton("单个方向")
        self.batch_mode_radio = QRadioButton("批处理 (双方向)")
        self.single_mode_radio.setToolTip("仅处理单个方向的条纹（垂直或水平）。\n您需要手动指定图像文件或文件夹。")
        self.batch_mode_radio.setToolTip("自动处理垂直和水平两个方向的条纹。\n您需要分别指定包含垂直和水平条纹的文件夹。")
        self.mode_group.addButton(self.single_mode_radio, 0)
        self.mode_group.addButton(self.batch_mode_radio, 1)
        self.single_mode_radio.setChecked(True)
        self.mode_group.buttonClicked.connect(self.on_mode_changed)
        mode_layout.addWidget(self.single_mode_radio)
        mode_layout.addWidget(self.batch_mode_radio)
        layout.addWidget(mode_group)
        
        # 相移参数组
        phase_group = QGroupBox("相移参数设置")
        phase_layout = QVBoxLayout(phase_group)
        
        # 相移步数
        steps_layout = QHBoxLayout()
        steps_layout.addWidget(QLabel("相移步数:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setMinimum(3)
        self.steps_spin.setMaximum(20)
        self.steps_spin.setValue(4)
        self.steps_spin.valueChanged.connect(self.on_steps_changed)
        self.steps_spin.setToolTip("设置计算包裹相位所用的相移步数(N)。\n这个值必须与您拍摄的相移图像数量一致。")
        steps_layout.addWidget(self.steps_spin)
        phase_layout.addLayout(steps_layout)
        
        # 条纹方向
        direction_layout = QHBoxLayout()
        direction_layout.addWidget(QLabel("条纹方向:"))
        direction_group = QWidget()
        direction_group_layout = QHBoxLayout(direction_group)
        direction_group_layout.setContentsMargins(0, 0, 0, 0)
        
        self.direction_group = QButtonGroup(self)
        self.horizontal_radio = QRadioButton("水平条纹")
        self.vertical_radio = QRadioButton("垂直条纹")
        self.horizontal_radio.setToolTip("表示加载的图像是水平条纹，用于计算垂直方向的包裹相位。")
        self.vertical_radio.setToolTip("表示加载的图像是垂直条纹，用于计算水平方向的包裹相位。")
        self.direction_group.addButton(self.horizontal_radio, 0)
        self.direction_group.addButton(self.vertical_radio, 1)
        self.vertical_radio.setChecked(True)  # 默认选择垂直条纹
        
        direction_group_layout.addWidget(self.horizontal_radio)
        direction_group_layout.addWidget(self.vertical_radio)
        direction_layout.addWidget(direction_group)
        phase_layout.addLayout(direction_layout)
        
        layout.addWidget(phase_group)
        
        # 图像输入堆叠窗口，用于模式切换
        self.input_stack = QWidget()
        self.input_stack_layout = QVBoxLayout(self.input_stack)
        self.input_stack_layout.setContentsMargins(0, 0, 0, 0)
        
        # 单模式输入面板
        self.single_input_panel = self.create_single_input_panel()
        self.input_stack_layout.addWidget(self.single_input_panel)

        # 批处理模式输入面板
        self.batch_input_panel = self.create_batch_input_panel()
        self.input_stack_layout.addWidget(self.batch_input_panel)
        self.batch_input_panel.setVisible(False)
        
        layout.addWidget(self.input_stack)
        
        # 预览区域
        self.preview_widget = PreviewWidget()
        layout.addWidget(self.preview_widget)
        
        # 设置滚动区域以防止面板太长
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(panel)
        
        return scroll_area
        
    def create_single_input_panel(self):
        """创建单方向输入面板"""
        input_group = QGroupBox("图像输入")
        input_layout = QVBoxLayout(input_group)

        # 输入模式选择
        input_mode_layout = QHBoxLayout()
        self.input_mode_group = QButtonGroup(self)
        self.files_radio = QRadioButton("选择文件")
        self.folder_radio = QRadioButton("选择文件夹")
        self.files_radio.setToolTip("手动选择多张相移图像文件。")
        self.folder_radio.setToolTip("选择一个包含所有相移图像的文件夹。\n程序会尝试根据文件名（如 I1.png, I2.png...）自动排序。")
        self.input_mode_group.addButton(self.files_radio, 0)
        self.input_mode_group.addButton(self.folder_radio, 1)
        self.files_radio.setChecked(True)

        input_mode_layout.addWidget(self.files_radio)
        input_mode_layout.addWidget(self.folder_radio)
        input_layout.addLayout(input_mode_layout)

        # 文件选择器
        file_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("选择图像文件...")
        self.file_path_edit.setReadOnly(True)
        self.browse_button = QPushButton("浏览...")
        self.browse_button.setToolTip("点击浏览文件或文件夹。\n选择文件夹时，程序会根据方向和步数自动加载 I1, I2... 或 I(N+1), I(N+2)... 等文件。")
        self.browse_button.clicked.connect(self.on_browse_clicked)
        file_layout.addWidget(self.file_path_edit, 3)
        file_layout.addWidget(self.browse_button, 1)
        input_layout.addLayout(file_layout)

        # 图像处理选项
        process_layout = QHBoxLayout()
        
        # 尺寸调整方法
        method_layout = QVBoxLayout()
        method_layout.addWidget(QLabel("尺寸调整方法:"))
        self.method_combo = QComboBox()
        self.method_combo.addItem("裁剪", "crop")
        self.method_combo.addItem("缩放", "resize")
        self.method_combo.setToolTip("当加载的图像尺寸不一时，选择处理方式：\n- 裁剪: 以最小图像的尺寸为基准，裁剪其他图像的中心区域。\n- 缩放: 将所有图像缩放到统一尺寸（可能影响精度）。")
        method_layout.addWidget(self.method_combo)
        process_layout.addLayout(method_layout)
        
        # 自定义尺寸
        size_layout = QVBoxLayout()
        size_layout.addWidget(QLabel("自定义尺寸 (可选):"))
        size_widget = QWidget()
        size_widget_layout = QHBoxLayout(size_widget)
        size_widget_layout.setContentsMargins(0, 0, 0, 0)
        
        self.width_spin = QSpinBox()
        self.width_spin.setToolTip("设置统一的图像宽度。\n设为0表示自动（以第一张图为准）。")
        self.width_spin.setMinimum(0)
        self.width_spin.setMaximum(9999)
        self.width_spin.setValue(0)
        self.width_spin.setSpecialValueText("自动")
        
        self.height_spin = QSpinBox()
        self.height_spin.setToolTip("设置统一的图像高度。\n设为0表示自动（以第一张图为准）。")
        self.height_spin.setMinimum(0)
        self.height_spin.setMaximum(9999)
        self.height_spin.setValue(0)
        self.height_spin.setSpecialValueText("自动")
        
        size_widget_layout.addWidget(QLabel("宽:"))
        size_widget_layout.addWidget(self.width_spin)
        size_widget_layout.addWidget(QLabel("高:"))
        size_widget_layout.addWidget(self.height_spin)
        
        size_layout.addWidget(size_widget)
        process_layout.addLayout(size_layout)
        
        input_layout.addLayout(process_layout)
        
        # 输出路径
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出目录:"))
        self.output_path_edit = QLineEdit("results")
        self.output_browse_button = QPushButton("浏览...")
        self.output_browse_button.clicked.connect(self.on_output_browse_clicked)
        self.output_path_edit.setToolTip("指定保存包裹相位结果的文件夹。\n结果会保存在该目录下的 'wrapped_phase_horizontal' 或 'wrapped_phase_vertical' 子文件夹中。")
        
        output_layout.addWidget(self.output_path_edit, 3)
        output_layout.addWidget(self.output_browse_button, 1)
        input_layout.addLayout(output_layout)
        
        # 加载和处理按钮
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("加载图像")
        self.load_button.clicked.connect(self.on_load_clicked)
        self.process_button = QPushButton("计算包裹相位")
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.on_process_clicked)
        self.load_button.setToolTip("根据当前设置加载指定的图像文件或文件夹。")
        self.process_button.setToolTip("对已加载的图像执行包裹相位计算。")
        
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.process_button)
        input_layout.addLayout(button_layout)

        return input_group

    def create_batch_input_panel(self):
        """创建批处理输入面板"""
        input_group = QGroupBox("批处理图像输入")
        input_layout = QVBoxLayout(input_group)

        # 垂直条纹输入
        v_layout = QHBoxLayout()
        self.v_folder_edit = QLineEdit()
        self.v_folder_edit.setPlaceholderText("选择垂直条纹图像文件夹...")
        self.v_browse_button = QPushButton("垂直条纹...")
        self.v_folder_edit.setToolTip("指定包含垂直条纹相移图像的文件夹。")
        self.v_browse_button.clicked.connect(lambda: self.on_batch_browse_clicked('vertical'))
        v_layout.addWidget(self.v_folder_edit, 3)
        v_layout.addWidget(self.v_browse_button, 1)
        input_layout.addLayout(v_layout)

        # 水平条纹输入
        h_layout = QHBoxLayout()
        self.h_folder_edit = QLineEdit()
        self.h_folder_edit.setPlaceholderText("选择水平条纹图像文件夹...")
        self.h_browse_button = QPushButton("水平条纹...")
        self.h_folder_edit.setToolTip("指定包含水平条纹相移图像的文件夹。")
        self.h_browse_button.clicked.connect(lambda: self.on_batch_browse_clicked('horizontal'))
        h_layout.addWidget(self.h_folder_edit, 3)
        h_layout.addWidget(self.h_browse_button, 1)
        input_layout.addLayout(h_layout)

        # 输出路径
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出目录:"))
        self.batch_output_edit = QLineEdit("results")
        self.batch_output_browse_button = QPushButton("浏览...")
        self.batch_output_edit.setToolTip("指定保存所有包裹相位结果的根文件夹。\n结果会分别保存在该目录下的子文件夹中。")
        self.batch_output_browse_button.clicked.connect(self.on_output_browse_clicked)
        output_layout.addWidget(self.batch_output_edit, 3)
        output_layout.addWidget(self.batch_output_browse_button, 1)
        input_layout.addLayout(output_layout)
        
        # 加载和处理按钮
        button_layout = QHBoxLayout()
        self.batch_load_button = QPushButton("加载所有图像")
        self.batch_load_button.clicked.connect(self.on_load_clicked)
        self.batch_process_button = QPushButton("计算所有包裹相位")
        self.batch_process_button.setEnabled(False)
        self.batch_process_button.clicked.connect(self.on_process_clicked)
        self.batch_load_button.setToolTip("加载您在上面指定的垂直和水平条纹图像文件夹。")
        self.batch_process_button.setToolTip("对所有已加载的图像执行包裹相位计算。")
        
        button_layout.addWidget(self.batch_load_button)
        button_layout.addWidget(self.batch_process_button)
        input_layout.addLayout(button_layout)

        return input_group

    def create_result_panel(self):
        """创建结果显示面板"""
        panel = QTabWidget()
        
        # 结果选择
        self.result_selector = QComboBox()
        self.result_selector.addItem("垂直条纹结果", "v")
        self.result_selector.addItem("水平条纹结果", "h")
        self.result_selector.setVisible(False) # 默认隐藏
        self.result_selector.currentIndexChanged.connect(self.on_result_selected)

        # 将结果选择器和选项卡放在一个布局中
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        result_layout.addWidget(self.result_selector)
        
        # 原始包裹相位选项卡
        self.tabs = QTabWidget()
        original_tab = QWidget()
        original_layout = QVBoxLayout(original_tab)
        self.original_image = PhaseImageWidget()
        original_layout.addWidget(self.original_image)
        self.tabs.addTab(original_tab, "包裹相位 (原始)")
        
        # 彩色包裹相位选项卡
        color_tab = QWidget()
        color_layout = QVBoxLayout(color_tab)
        self.color_image = PhaseImageWidget()
        color_layout.addWidget(self.color_image)
        self.tabs.addTab(color_tab, "包裹相位 (彩色)")
        
        # 直方图均衡化选项卡
        equalized_tab = QWidget()
        equalized_layout = QVBoxLayout(equalized_tab)
        self.equalized_image = PhaseImageWidget()
        equalized_layout.addWidget(self.equalized_image)
        self.tabs.addTab(equalized_tab, "包裹相位 (均衡化)")
        
        # 信息选项卡
        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)
        self.info_label = QLabel("暂无结果数据")
        self.info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.info_label.setWordWrap(True)
        info_scroll = QScrollArea()
        info_scroll.setWidget(self.info_label)
        info_scroll.setWidgetResizable(True)
        info_layout.addWidget(info_scroll)
        self.tabs.addTab(info_tab, "结果信息")

        result_layout.addWidget(self.tabs)
        panel.addTab(result_widget, "计算结果")
        
        return panel
        
    def on_mode_changed(self):
        is_batch = self.batch_mode_radio.isChecked()
        self.single_input_panel.setVisible(not is_batch)
        self.batch_input_panel.setVisible(is_batch)
        # 批处理模式下，方向选择无效
        self.horizontal_radio.setEnabled(not is_batch)
        self.vertical_radio.setEnabled(not is_batch)
        self.result_selector.setVisible(is_batch)

    def on_result_selected(self):
        """切换显示不同方向的结果"""
        self.display_results(self.v_phase_result, self.h_phase_result, self.last_prefix)

    def on_steps_changed(self, value):
        """相移步数改变时的回调"""
        self.wrapped_phase.n = value
        
    def on_browse_clicked(self):
        """浏览按钮点击回调"""
        if self.files_radio.isChecked():
            # 选择多个文件
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "选择相移图像文件",
                "",
                "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*.*)"
            )
            
            if files:
                # 只显示文件名，不显示完整路径
                file_names = [os.path.basename(f) for f in files]
                self.file_path_edit.setText(", ".join(file_names))
                self.file_path_edit.setToolTip("\n".join(files))
                # 保存完整路径
                self.selected_files = files
        else:
            # 选择文件夹
            folder = QFileDialog.getExistingDirectory(
                self,
                "选择包含相移图像的文件夹",
                ""
            )
            
            if folder:
                self.file_path_edit.setText(folder)
                self.file_path_edit.setToolTip(folder)
                self.selected_files = folder
                
    def on_batch_browse_clicked(self, direction):
        """批处理模式下的浏览按钮"""
        folder = QFileDialog.getExistingDirectory(self, f"选择{direction}条纹图像文件夹")
        if folder:
            if direction == 'vertical':
                self.v_folder_edit.setText(folder)
            else:
                self.h_folder_edit.setText(folder)

    def on_output_browse_clicked(self):
        """输出目录浏览按钮点击回调"""
        is_batch = self.batch_mode_radio.isChecked()
        current_path = self.batch_output_edit.text() if is_batch else self.output_path_edit.text()
        
        folder = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录",
            current_path
        )
        
        if folder:
            if is_batch:
                self.batch_output_edit.setText(folder)
            else:
                self.output_path_edit.setText(folder)
            
    def on_load_clicked(self):
        """加载图像按钮点击回调"""
        if self.batch_mode_radio.isChecked():
            self.load_batch_images()
        else:
            self.load_single_images()

    def load_single_images(self):
        """加载单方向图像"""
        # 检查是否已选择文件
        if not hasattr(self, 'selected_files') or not self.selected_files:
            QMessageBox.warning(self, "警告", "请先选择图像文件或文件夹")
            return
            
        # 获取方向
        direction = "vertical" if self.vertical_radio.isChecked() else "horizontal"
        
        # 获取尺寸调整方法
        method = self.method_combo.currentData()
        
        # 获取目标尺寸
        width = self.width_spin.value()
        height = self.height_spin.value()
        target_size = None
        if width > 0 and height > 0:
            target_size = (height, width)  # OpenCV使用(height, width)顺序
            
        # 获取相移步数
        steps = self.steps_spin.value()
        self.wrapped_phase.n = steps
        
        try:
            self.status_bar.showMessage("正在加载图像...")
            self.progress_bar.setValue(10)
            self.progress_bar.setVisible(True)
            
            # 加载图像
            if isinstance(self.selected_files, list):
                # 从文件列表加载
                image_paths = self.selected_files
                if len(image_paths) < 3:
                    raise ValueError(f"至少需要3幅相移图像，但只提供了{len(image_paths)}幅")
                elif len(image_paths) != steps:
                    QMessageBox.warning(
                        self, 
                        "警告", 
                        f"指定的相移步数为{steps}，但提供了{len(image_paths)}幅图像\n"
                        f"将使用提供的{len(image_paths)}幅图像作为实际相移步数"
                    )
                    steps = len(image_paths)
                    self.wrapped_phase.n = steps
                    self.steps_spin.setValue(steps)
            else:
                # 从文件夹加载
                folder_path = self.selected_files
                
                # 查找图像文件
                image_paths = []
                patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
                
                # 首先尝试按I1, I2, I3...命名规则查找
                if direction == "vertical":
                    target_files = [f"I{i+1}.png" for i in range(steps)]
                else: # direction == "horizontal"
                    target_files = [f"I{i+steps+1}.png" for i in range(steps)]
                
                for file_name in target_files:
                    file_path = os.path.join(folder_path, file_name)
                    if os.path.exists(file_path):
                        image_paths.append(file_path)
                
                # 如果按上述规则找不到足够的图像，尝试其他命名规则
                if len(image_paths) < steps:
                    # 清空当前列表，重新查找所有图像文件
                    image_paths = []
                    for pattern in patterns:
                        image_paths.extend(glob.glob(os.path.join(folder_path, pattern)))
                    
                    # 对文件进行排序
                    def extract_number(path):
                        import re
                        match = re.search(r'(\d+)', os.path.basename(path))
                        if match:
                            return int(match.group(1))
                        return 0
                    
                    image_paths.sort(key=extract_number)
                
                # 检查是否找到足够的图像
                if len(image_paths) < 3:
                    raise ValueError(f"在文件夹中至少需要3幅相移图像，但只找到{len(image_paths)}幅")
                elif len(image_paths) != steps:
                    result = QMessageBox.question(
                        self, 
                        "调整相移步数", 
                        f"找到{len(image_paths)}幅图像，但指定的相移步数为{steps}\n"
                        f"是否要将相移步数调整为{len(image_paths)}？",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    
                    if result == QMessageBox.Yes:
                        steps = len(image_paths)
                        self.wrapped_phase.n = steps
                        self.steps_spin.setValue(steps)
                    elif len(image_paths) > steps:
                        # 如果找到的图像太多，截取前steps个
                        image_paths = image_paths[:steps]
            
            self.progress_bar.setValue(50)
            
            # 使用wrapped_phase中的getImageData方法加载图像
            self.loaded_images = self.wrapped_phase.getImageData(
                image_paths, 
                direction, 
                target_size, 
                method,
                n=steps
            )
            
            # 检查图像是否成功加载
            valid_images = [img for img in self.loaded_images if img is not None]
            if len(valid_images) < 3:
                raise ValueError(f"至少需要3幅有效相移图像，但只加载了{len(valid_images)}幅")
                
            self.progress_bar.setValue(80)
            
            # 更新预览
            self.preview_widget.set_images(self.loaded_images)
            
            # 启用处理按钮
            self.process_button.setEnabled(True)
            
            self.progress_bar.setValue(100)
            self.status_bar.showMessage(f"成功加载{len(valid_images)}幅图像")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像失败: {str(e)}")
            self.status_bar.showMessage("加载图像失败")
        finally:
            self.progress_bar.setVisible(False)
            
    def load_batch_images(self):
        """加载批处理模式下的所有图像"""
        v_folder = self.v_folder_edit.text()
        h_folder = self.h_folder_edit.text()

        if not v_folder and not h_folder:
            QMessageBox.warning(self, "警告", "请至少选择一个方向的图像文件夹")
            return

        self.loaded_images_v = []
        self.loaded_images_h = []
        all_loaded_for_preview = []
        
        steps = self.steps_spin.value()
        self.wrapped_phase.n = steps

        try:
            self.status_bar.showMessage("正在加载所有图像...")
            # 加载垂直条纹
            if v_folder:
                self.loaded_images_v = self.wrapped_phase.getImageData(None, "vertical", None, "crop", n=steps)
                all_loaded_for_preview.extend(self.loaded_images_v)
            # 加载水平条纹
            if h_folder:
                self.loaded_images_h = self.wrapped_phase.getImageData(None, "horizontal", None, "crop", n=steps)
                all_loaded_for_preview.extend(self.loaded_images_h)

            self.preview_widget.set_images(all_loaded_for_preview)
            self.batch_process_button.setEnabled(True)
            self.status_bar.showMessage(f"加载完成: {len(self.loaded_images_v)}幅垂直条纹, {len(self.loaded_images_h)}幅水平条纹")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载批处理图像失败: {str(e)}")
            self.status_bar.showMessage("加载图像失败")

    def on_process_clicked(self):
        """处理按钮点击回调"""
        if self.batch_mode_radio.isChecked():
            # 批处理
            if (not self.loaded_images_v or len(self.loaded_images_v) < 3) and \
               (not self.loaded_images_h or len(self.loaded_images_h) < 3):
                QMessageBox.warning(self, "警告", "请先加载至少一个方向的有效图像 (不少于3幅)")
                return
            output_dir = self.batch_output_edit.text() or "results"
        else:
            # 单处理
            if not self.loaded_images or len(self.loaded_images) < 3:
                QMessageBox.warning(self, "警告", "请先加载至少3幅有效的相移图像")
                return
            output_dir = self.output_path_edit.text() or "results"

        # 确保输出目录存在
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"创建输出目录失败: {str(e)}")
                return
                
        # 启动处理线程
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage("正在计算包裹相位...")
        
        if self.batch_mode_radio.isChecked():
            self.batch_process_button.setEnabled(False)
            self.processing_thread = ImageProcessingThread(self.wrapped_phase, self.loaded_images_v, self.loaded_images_h)
        else:
            self.process_button.setEnabled(False)
            direction = "vertical" if self.vertical_radio.isChecked() else "horizontal"
            v_imgs = self.loaded_images if direction == 'vertical' else []
            h_imgs = self.loaded_images if direction == 'horizontal' else []
            self.processing_thread = ImageProcessingThread(self.wrapped_phase, v_imgs, h_imgs)
        
        # 连接信号
        self.processing_thread.progress_update.connect(self.update_progress)
        self.processing_thread.processing_complete.connect(self.on_processing_complete)
        self.processing_thread.processing_error.connect(self.on_processing_error)
        
        # 启动线程
        self.processing_thread.start()
            
    def update_progress(self, value, message):
        """更新进度条和状态栏"""
        self.progress_bar.setValue(value)
        self.status_bar.showMessage(message)
            
    def on_processing_complete(self, v_phase, h_phase, prefix_base):
        """处理完成回调"""
        self.v_phase_result = v_phase
        self.h_phase_result = h_phase
        self.last_prefix = prefix_base
        
        output_dir = self.batch_output_edit.text() if self.batch_mode_radio.isChecked() else self.output_path_edit.text()
        output_dir = output_dir or "results"

        # 保存结果
        saved_paths = []
        try:
            if v_phase is not None:
                path = self.wrapped_phase.save_wrapped_phase(v_phase, output_dir, "vertical_fringe_" + prefix_base, "vertical")
                saved_paths.append(f"垂直条紋结果保存至: {os.path.abspath(path)}")
            if h_phase is not None:
                path = self.wrapped_phase.save_wrapped_phase(h_phase, output_dir, "horizontal_fringe_" + prefix_base, "horizontal")
                saved_paths.append(f"水平条紋结果保存至: {os.path.abspath(path)}")
        except Exception as e:
            QMessageBox.warning(self, "警告", f"保存结果时出错: {str(e)}")

        self.display_results(v_phase, h_phase, prefix_base)
            
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("包裹相位计算完成")
        self.process_button.setEnabled(True)
        self.batch_process_button.setEnabled(True)

    def display_results(self, v_phase, h_phase, prefix_base):
        """根据选择更新UI显示"""
        is_batch = self.batch_mode_radio.isChecked()
        current_selection = self.result_selector.currentData() if is_batch else None
        
        wrapped_phase = None
        direction = 'vertical'

        if is_batch:
            if current_selection == 'v':
                wrapped_phase = v_phase
                direction = 'vertical'
            else: # 'h'
                wrapped_phase = h_phase
                direction = 'horizontal'
        else:
            wrapped_phase = v_phase if v_phase is not None else h_phase
            direction = "vertical" if self.vertical_radio.isChecked() else "horizontal"

        if wrapped_phase is None:
            self.original_image.image_label.setText("无此方向结果")
            self.color_image.image_label.setText("无此方向结果")
            self.equalized_image.image_label.setText("无此方向结果")
            self.info_label.setText("无此方向的结果数据。")
            return

        # 更新UI显示
        try:
            # 将相位值缩放到[0,255]范围用于显示
            phase_scaled = (wrapped_phase * 255 / (2*np.pi)).astype(np.uint8)
            
            # 原始缩放视图
            self.original_image.set_image(phase_scaled)
            
            # 应用伪彩色映射以增强可视化效果
            phase_color = cv.applyColorMap(phase_scaled, cv.COLORMAP_JET)
            self.color_image.set_image(phase_color)
            
            # 应用直方图均衡化以增强对比度
            phase_eq = cv.equalizeHist(phase_scaled)
            self.equalized_image.set_image(cv.applyColorMap(phase_eq, cv.COLORMAP_JET))

            # 更新信息标签
            min_phase = np.min(wrapped_phase)
            max_phase = np.max(wrapped_phase)
            mean_phase = np.mean(wrapped_phase)
            std_phase = np.std(wrapped_phase)
            
            # 确定子目录
            sub_dir_map = {"vertical": "wrapped_phase_horizontal", "horizontal": "wrapped_phase_vertical"}
            output_dir = self.batch_output_edit.text() if is_batch else self.output_path_edit.text()
            full_path = os.path.join(output_dir or "results", sub_dir_map[direction])

            info_text = f"""<h3>包裹相位计算结果</h3>
            <p><b>相移步数:</b> {self.wrapped_phase.n}</p>
            <p><b>条纹方向:</b> {'垂直条纹 (用于水平方向解包裹)' if direction == 'vertical' else '水平条纹 (用于垂直方向解包裹)'}</p>
            <p><b>图像尺寸:</b> {wrapped_phase.shape[1]}×{wrapped_phase.shape[0]} (宽×高)</p>
            <p><b>相位范围:</b> [{min_phase:.6f}, {max_phase:.6f}] rad</p>
            <p><b>相位均值:</b> {mean_phase:.6f} rad</p>
            <p><b>相位标准差:</b> {std_phase:.6f} rad</p>
            <p><b>结果保存位置:</b> {os.path.abspath(full_path)}</p>
            """
            
            self.info_label.setText(info_text)
            
        except Exception as e:
            QMessageBox.warning(self, "警告", f"显示结果时出错: {str(e)}")

    def on_processing_error(self, error_message):
        """处理错误回调"""
        QMessageBox.critical(self, "错误", f"计算包裹相位时出错: {error_message}")
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("计算包裹相位失败")
        self.process_button.setEnabled(True)
        self.batch_process_button.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    window = WrappedPhaseUI()
    window.show()
    sys.exit(app.exec())
    
if __name__ == "__main__":
    main() 