import sys
import os
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QSpinBox, QLineEdit, 
                               QPushButton, QFileDialog, QRadioButton, QButtonGroup, 
                               QGroupBox, QTabWidget, QSplitter, QMessageBox, 
                               QProgressBar, QScrollArea, QCheckBox, QComboBox, QDialog, QTextBrowser)
from PySide6.QtGui import QPixmap, QImage, QColor, QPalette, QPainter, QPen
from PySide6.QtCore import Qt, Signal, Slot, QThread, QObject
import cv2 as cv

# 导入核心处理逻辑
import phase_unwrapper as pu

class ProcessingThread(QThread):
    """用于后台处理相位解包裹的线程"""
    progress_update = Signal(int, str)
    processing_complete = Signal(dict)
    processing_error = Signal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            results = {}
            h_unwrapped_results, v_unwrapped_results = [], []
            h_basenames, v_basenames = [], []

            # 封装 process_unwrapping 调用以捕获输出
            def process_and_report(direction, wrapped_input, graycode_folder, n, adaptive, show=False):
                self.progress_update.emit(10, f"开始处理 {direction} 方向...")
                
                # 加载格雷码图像
                graycode_images = []
                image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
                graycode_files = sorted([f for f in os.listdir(graycode_folder) if f.lower().endswith(image_extensions)])
                if len(graycode_files) < n:
                    raise ValueError(f"在 {graycode_folder} 中需要至少{n}张格雷码图像，但只找到{len(graycode_files)}张")

                for i, fname in enumerate(graycode_files[:n]):
                    self.progress_update.emit(10 + int(i/n * 20), f"加载格雷码图像 {i+1}/{n}...")
                    path = os.path.join(graycode_folder, fname)
                    img = cv.imread(path, -1)
                    if img is None: raise ValueError(f"无法读取格雷码图像 {path}")
                    graycode_images.append(img)
                
                # 加载包裹相位图像
                wrapped_phase_paths = []
                if os.path.isdir(wrapped_input):
                    self.progress_update.emit(25, f"在 {direction} 文件夹中查找 'wrapped_phase_equalized' 文件...")
                    # 增强文件查找逻辑：递归搜索子目录
                    search_keyword = "wrapped_phase_equalized"
                    found_files = []
                    for root, _, files in os.walk(wrapped_input):
                        for file in files:
                            if search_keyword in file.lower() and file.lower().endswith(image_extensions):
                                found_files.append(os.path.join(root, file))
                    
                    # 如果直接在顶层目录找不到，但在子目录找到了，就使用子目录的结果
                    if found_files:
                        wrapped_phase_paths = sorted(found_files)
                    else: # 保持原有的顶层目录查找逻辑作为后备
                        files = sorted([
                            f for f in os.listdir(wrapped_input) 
                            if search_keyword in f.lower() and f.lower().endswith(image_extensions)
                        ])
                        for f in files: wrapped_phase_paths.append(os.path.join(wrapped_input, f))

                elif os.path.isfile(wrapped_input):
                    wrapped_phase_paths.append(wrapped_input)

                if not wrapped_phase_paths:
                    raise ValueError(f"在 {wrapped_input} 中找不到有效的包裹相位图像（或 'wrapped_phase_equalized' 文件）。")

                unwrapper = pu.PhaseUnwrapper(n=n, direction=direction)
                
                unwrapped_phases = []
                basenames = []

                for i, path in enumerate(wrapped_phase_paths):
                    self.progress_update.emit(30 + int(i/len(wrapped_phase_paths) * 60), f"正在处理 {os.path.basename(path)}...")
                    wrapped_phase = cv.imread(path, -1)
                    if wrapped_phase is None: continue
                    
                    if wrapped_phase.dtype != np.float32:
                        if wrapped_phase.dtype == np.uint8:
                            wrapped_phase = wrapped_phase.astype(np.float32) * (2 * np.pi / 255)
                        elif wrapped_phase.dtype == np.uint16:
                            wrapped_phase = wrapped_phase.astype(np.float32) * (2 * np.pi / 65535)
                        else:
                            wrapped_phase = wrapped_phase.astype(np.float32)
                    
                    basename = os.path.splitext(os.path.basename(path))[0]
                    
                    smoothed_phase = unwrapper.unwrap_phase(
                        wrapped_phase,
                        graycode_images,
                        adaptive_threshold=adaptive,
                        show_results=False, # GUI控制显示
                        basename=basename
                    )
                    unwrapped_phases.append(smoothed_phase)
                    basenames.append(basename)

                self.progress_update.emit(95, f"{direction} 方向处理完成。")
                return unwrapped_phases, basenames

            # --- 根据模式执行解包裹 ---
            mode = self.params['mode']
            n_gray = self.params['n_graycodes']
            adaptive = self.params['adaptive_threshold']

            if mode in ['horizontal', 'both']:
                h_unwrapped_results, h_basenames = process_and_report(
                    "horizontal", self.params['h_wrapped_input'], self.params['v_graycode_folder'], n_gray, adaptive
                )
                results['h_unwrapped'] = h_unwrapped_results
                results['h_basenames'] = h_basenames

            if mode in ['vertical', 'both']:
                v_unwrapped_results, v_basenames = process_and_report(
                    "vertical", self.params['v_wrapped_input'], self.params['h_graycode_folder'], n_gray, adaptive
                )
                results['v_unwrapped'] = v_unwrapped_results
                results['v_basenames'] = v_basenames

            # --- 合并结果 ---
            if mode == 'both' and h_unwrapped_results and v_unwrapped_results:
                self.progress_update.emit(98, "合并水平和垂直结果...")
                if len(h_unwrapped_results) == len(v_unwrapped_results):
                    combined_results = []
                    for i in range(len(h_unwrapped_results)):
                        combined_img = pu.generate_combined_phase_image(
                            h_unwrapped_results[i], v_unwrapped_results[i], output_path=None
                        )
                        combined_results.append(combined_img)
                    results['combined'] = combined_results
            
            self.progress_update.emit(100, "全部处理完成!")
            self.processing_complete.emit(results)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.processing_error.emit(str(e))

class InteractiveImage(QLabel):
    """可交互的图像显示控件，用于显示相位值"""
    phase_info_updated = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.setText("无结果图像")

        self.pixmap = None
        self.display_pixmap = None
        self.data_maps = {} # e.g. {'h_phase': np.array, 'v_phase': np.array}
        self.setMouseTracking(True)
        self.last_pos = None
        self.hover_pos = None

    def set_image(self, cv_img, data_maps={}):
        if cv_img is None:
            self.clear()
            self.setText("无结果图像")
            self.data_maps = {}
            self.pixmap = None
            self.display_pixmap = None
            return

        self.data_maps = data_maps
        h, w = cv_img.shape[:2]
        
        # 转换为 QImage
        if len(cv_img.shape) == 2:
            img_format = QImage.Format_Grayscale8
            bytes_per_line = w
        else: # 3 channels
            cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
            img_format = QImage.Format_RGB888
            bytes_per_line = 3 * w
        
        q_img = QImage(cv_img.data, w, h, bytes_per_line, img_format)
        self.pixmap = QPixmap.fromImage(q_img)
        self.update_display()
        
    def update_display(self):
        if not self.pixmap:
            return
        self.display_pixmap = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(self.display_pixmap)

    def mouseMoveEvent(self, event):
        self.hover_pos = event.pos()
        self.update() # Trigger repaint

        if not self.pixmap or not self.display_pixmap:
            self.phase_info_updated.emit("无有效图像数据")
            return

        # 转换坐标: 控件坐标 -> 缩放后图像坐标 -> 原始图像坐标
        widget_pos = event.pos()
        
        # 考虑到图像可能不是填满控件的，需要计算偏移
        pixmap_size = self.display_pixmap.size()
        widget_size = self.size()
        x_offset = (widget_size.width() - pixmap_size.width()) / 2
        y_offset = (widget_size.height() - pixmap_size.height()) / 2
        
        pixmap_x = widget_pos.x() - x_offset
        pixmap_y = widget_pos.y() - y_offset

        if not (0 <= pixmap_x < pixmap_size.width() and 0 <= pixmap_y < pixmap_size.height()):
            self.phase_info_updated.emit("将光标移入图像以查看信息")
            return
            
        scale_x = self.pixmap.width() / self.display_pixmap.width()
        scale_y = self.pixmap.height() / self.display_pixmap.height()
        
        orig_x = int(pixmap_x * scale_x)
        orig_y = int(pixmap_y * scale_y)

        # 确保坐标在数据范围内
        if not (self.data_maps and 'h_phase' in self.data_maps and self.data_maps['h_phase'] is not None and \
              0 <= orig_y < self.data_maps['h_phase'].shape[0] and 0 <= orig_x < self.data_maps['h_phase'].shape[1]) and \
           not (self.data_maps and 'v_phase' in self.data_maps and self.data_maps['v_phase'] is not None and \
              0 <= orig_y < self.data_maps['v_phase'].shape[0] and 0 <= orig_x < self.data_maps['v_phase'].shape[1]):
            self.phase_info_updated.emit("坐标超出数据范围")
            return

        # 获取相位信息
        info = f"坐标: ({orig_x}, {orig_y}) | "
        if 'h_phase' in self.data_maps and self.data_maps['h_phase'] is not None:
            info += f"水平相位: {self.data_maps['h_phase'][orig_y, orig_x]:.4f} | "
        if 'v_phase' in self.data_maps and self.data_maps['v_phase'] is not None:
            info += f"垂直相位: {self.data_maps['v_phase'][orig_y, orig_x]:.4f}"

        self.phase_info_updated.emit(info.strip().strip('|').strip())

    def leaveEvent(self, event):
        self.hover_pos = None
        self.update() # Trigger repaint
        self.phase_info_updated.emit("将光标移入图像以查看信息")

    def mousePressEvent(self, event):
        # 点击事件现在可以用于其他功能，例如设置一个固定标记
        pass
        
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.hover_pos or not self.display_pixmap:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        pen = QPen(QColor(255, 255, 0, 150))
        pen.setStyle(Qt.DashLine)
        pen.setWidth(1)
        painter.setPen(pen)

        # Draw crosshair lines
        painter.drawLine(self.hover_pos.x(), 0, self.hover_pos.x(), self.height())
        painter.drawLine(0, self.hover_pos.y(), self.width(), self.hover_pos.y())

    def resizeEvent(self, event):
        self.update_display()

class PhaseUnwrapperUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("相位解包裹工具")
        self.setMinimumSize(1200, 700)
        self._set_style()

        # 中央控件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 控制面板
        control_panel_scroll = QScrollArea()
        control_panel_scroll.setWidgetResizable(True)
        control_panel_scroll.setWidget(self._create_control_panel())
        
        # 结果面板
        self.results_panel = self._create_results_panel()
        
        splitter.addWidget(control_panel_scroll)
        splitter.addWidget(self.results_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        # 状态栏
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage("就绪")
        
        self.results_data = {}

    def _set_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QGroupBox {
                font-weight: bold; border: 1px solid #ccc;
                border-radius: 5px; margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }
            QPushButton {
                background-color: #e1e1e1; border: 1px solid #adadad;
                padding: 5px; border-radius: 3px;
            }
            QPushButton:hover { background-color: #cacaca; }
            QLineEdit, QSpinBox, QComboBox { padding: 3px; border: 1px solid #ccc; border-radius: 3px; }
            QTabWidget::pane { border-top: 1px solid #ccc; }
            QTabBar::tab { padding: 8px 20px; border: 1px solid #ccc; border-bottom: none; 
                           border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #f0f0f0; margin-bottom: -1px; }
            QTabBar::tab:!selected { background: #e1e1e1; }
        """)

    def _create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 模式选择
        mode_group = QGroupBox("1. 选择解包裹模式")
        mode_layout = QHBoxLayout(mode_group)
        self.mode_group = QButtonGroup(self)
        self.h_radio = QRadioButton("仅水平")
        self.v_radio = QRadioButton("仅垂直")
        self.both_radio = QRadioButton("双向合并")
        self.h_radio.setToolTip("只执行水平方向的解包裹。\n需要输入包裹相位图(来自垂直条纹)和垂直格雷码。")
        self.v_radio.setToolTip("只执行垂直方向的解包裹。\n需要输入包裹相位图(来自水平条纹)和水平格雷码。")
        self.both_radio.setToolTip("执行水平和垂直两个方向的解包裹，并生成最终的组合相位图。\n这是推荐的模式，可以得到最完整的结果。")
        self.mode_group.addButton(self.h_radio, 0)
        self.mode_group.addButton(self.v_radio, 1)
        self.mode_group.addButton(self.both_radio, 2)
        self.both_radio.setChecked(True)
        self.mode_group.buttonClicked.connect(self._on_mode_change)
        mode_layout.addWidget(self.h_radio)
        mode_layout.addWidget(self.v_radio)
        mode_layout.addWidget(self.both_radio)
        layout.addWidget(mode_group)

        # 路径输入
        path_group = QGroupBox("2. 设置输入路径")
        path_layout = QVBoxLayout(path_group)
        
        # 水平解包裹路径
        self.h_group = QGroupBox("水平方向解包裹 (需要垂直条纹)")
        self.h_group.setCheckable(True)
        self.h_group.setChecked(True)
        self.h_group.setToolTip("用于计算水平方向上的绝对相位。\n您需要提供由“垂直条纹”生成的包裹相位图，以及“垂直格雷码”图像。")
        h_layout = QVBoxLayout(self.h_group)
        h_layout.addLayout(self._create_wrapped_phase_input_selector("h_wrapped", "包裹相位:"))
        self.v_graycode_path = self._create_path_selector("垂直格雷码 (文件夹):", "v_gray", folder_only=True)
        h_layout.addLayout(self.v_graycode_path)
        path_layout.addWidget(self.h_group)

        # 垂直解包裹路径
        self.v_group = QGroupBox("垂直方向解包裹 (需要水平条纹)")
        self.v_group.setCheckable(True)
        self.v_group.setChecked(True)
        self.v_group.setToolTip("用于计算垂直方向上的绝对相位。\n您需要提供由“水平条纹”生成的包裹相位图，以及“水平格雷码”图像。")
        v_layout = QVBoxLayout(self.v_group)
        v_layout.addLayout(self._create_wrapped_phase_input_selector("v_wrapped", "包裹相位:"))
        self.h_graycode_path = self._create_path_selector("水平格雷码 (文件夹):", "h_gray", folder_only=True)
        v_layout.addLayout(self.h_graycode_path)
        path_layout.addWidget(self.v_group)

        layout.addWidget(path_group)

        # 参数设置
        param_group = QGroupBox("3. 参数设置")
        param_layout = QHBoxLayout(param_group)
        param_layout.addWidget(QLabel("格雷码位数:"))
        self.n_gray_spin = QSpinBox()
        self.n_gray_spin.setRange(3, 10)
        self.n_gray_spin.setValue(5)
        self.n_gray_spin.setToolTip("设置用于解包裹的格雷码图像的位数。\n这个数值必须与您加载的格雷码图像文件夹中的图像数量相匹配。")
        param_layout.addWidget(self.n_gray_spin)
        self.adaptive_check = QCheckBox("自适应阈值")
        self.adaptive_check.setToolTip("启用后，程序会根据图像内容自动计算二值化阈值。\n对于光照不均匀的图像，建议勾选此项。")
        param_layout.addWidget(self.adaptive_check)
        param_layout.addStretch()
        layout.addWidget(param_group)

        # 处理按钮
        self.process_button = QPushButton("开始解包裹")
        self.process_button.setFixedHeight(40)
        self.process_button.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.process_button.setToolTip("根据当前设置，开始执行相位解包裹流程。")
        self.process_button.clicked.connect(self._on_process)
        layout.addWidget(self.process_button)

        layout.addStretch()
        self._on_mode_change()
        return panel

    def _create_wrapped_phase_input_selector(self, name, label):
        """创建包含文件/文件夹选择的包裹相位输入控件"""
        main_layout = QVBoxLayout()
        
        radio_layout = QHBoxLayout()
        radio_group = QButtonGroup(self)
        file_radio = QRadioButton("文件")
        folder_radio = QRadioButton("文件夹")
        file_radio.setChecked(True)
        radio_group.addButton(file_radio, 0)
        radio_group.addButton(folder_radio, 1)
        
        radio_layout.addWidget(QLabel(label))
        radio_layout.addWidget(file_radio)
        radio_layout.addWidget(folder_radio)
        radio_layout.addStretch()
        setattr(self, f"{name}_input_mode_group", radio_group)
        
        file_radio.setToolTip("选择单个包裹相位图像文件进行处理。")
        folder_radio.setToolTip("选择一个包含多个包裹相位图像的文件夹进行批量处理。\n程序会自动查找文件名中带有 'wrapped_phase_equalized' 的文件。")

        main_layout.addLayout(radio_layout)
        
        path_layout = QHBoxLayout()
        edit = QLineEdit()
        edit.setReadOnly(True)
        setattr(self, f"{name}_edit", edit)
        button = QPushButton("浏览...")
        button.setToolTip(f"点击浏览以选择包裹相位的源文件或文件夹。")
        button.clicked.connect(lambda: self._browse_wrapped_phase(name))
        path_layout.addWidget(edit)
        path_layout.addWidget(button)
        main_layout.addLayout(path_layout)
        
        return main_layout

    def _create_path_selector(self, label, name, folder_only=False):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        edit = QLineEdit()
        edit.setReadOnly(True)
        setattr(self, f"{name}_edit", edit)
        button = QPushButton("浏览...")
        button.setToolTip(f"点击浏览以选择包含格雷码图像的文件夹。")
        button.clicked.connect(lambda: self._browse_graycode(name))
        layout.addWidget(edit)
        layout.addWidget(button)
        return layout

    def _create_results_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0,0,0,0)

        # 结果选择器
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("显示结果:"))
        self.result_selector = QComboBox()
        self.result_selector.addItem("水平解包裹结果", "h")
        self.result_selector.addItem("垂直解包裹结果", "v")
        self.result_selector.currentIndexChanged.connect(self._on_result_display_change)
        self.result_selector.setVisible(False) # 默认隐藏
        selector_layout.addWidget(self.result_selector)
        selector_layout.addStretch()
        layout.addLayout(selector_layout)

        self.tabs = QTabWidget()
        self.smoothed_tab = InteractiveImage()
        self.combined_tab = InteractiveImage()

        self.tabs.addTab(self.smoothed_tab, "平滑相位图")
        self.tabs.addTab(self.combined_tab, "组合相位图")

        layout.addWidget(self.tabs)

        info_group = QGroupBox("交互信息")
        info_layout = QVBoxLayout(info_group)
        self.info_label = QLabel("点击上方图像以显示此处的光标处相位信息。")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        info_scroll = QScrollArea()
        info_scroll.setWidget(self.info_label)
        info_scroll.setWidgetResizable(True)
        info_scroll.setFixedHeight(80)
        
        info_layout.addWidget(info_scroll)
        layout.addWidget(info_group)
        
        # 连接信号
        self.smoothed_tab.phase_info_updated.connect(self._update_info_label)
        self.combined_tab.phase_info_updated.connect(self._update_info_label)

        return panel

    @Slot()
    def _on_mode_change(self):
        mode_id = self.mode_group.checkedId()
        self.h_group.setVisible(mode_id in [0, 2])
        self.v_group.setVisible(mode_id in [1, 2])

    def _browse_wrapped_phase(self, name):
        edit_widget = getattr(self, f"{name}_edit")
        mode_group = getattr(self, f"{name}_input_mode_group")
        
        if mode_group.checkedId() == 0: # File
            path, _ = QFileDialog.getOpenFileName(self, "选择包裹相位文件", filter="图像文件 (*.png *.jpg *.jpeg *.bmp *.tif)")
        else: # Folder
            path = QFileDialog.getExistingDirectory(self, "选择包裹相位文件夹")

        if path:
            edit_widget.setText(path)

    def _browse_graycode(self, name):
        edit_widget = getattr(self, f"{name}_edit")
        path = QFileDialog.getExistingDirectory(self, "选择格雷码文件夹")
        if path:
            edit_widget.setText(path)

    @Slot()
    def _on_process(self):
        params = {}
        mode_id = self.mode_group.checkedId()
        if mode_id == 0: params['mode'] = 'horizontal'
        elif mode_id == 1: params['mode'] = 'vertical'
        else: params['mode'] = 'both'
        
        try:
            if params['mode'] in ['horizontal', 'both']:
                params['h_wrapped_input'] = self.h_wrapped_edit.text()
                params['v_graycode_folder'] = self.v_gray_edit.text()
                if not params['h_wrapped_input'] or not params['v_graycode_folder']:
                    raise ValueError("请为水平解包裹提供所有必需的路径。")
            if params['mode'] in ['vertical', 'both']:
                params['v_wrapped_input'] = self.v_wrapped_edit.text()
                params['h_graycode_folder'] = self.h_gray_edit.text()
                if not params['v_wrapped_input'] or not params['h_graycode_folder']:
                    raise ValueError("请为垂直解包裹提供所有必需的路径。")
        except ValueError as e:
            QMessageBox.warning(self, "路径错误", str(e))
            return

        params['n_graycodes'] = self.n_gray_spin.value()
        params['adaptive_threshold'] = self.adaptive_check.isChecked()

        self.process_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage("开始处理...")

        self.thread = ProcessingThread(params)
        self.thread.progress_update.connect(lambda v, m: self.status_bar.showMessage(f"进度: {v}% - {m}"))
        self.thread.processing_complete.connect(self._on_complete)
        self.thread.processing_error.connect(self._on_error)
        self.thread.start()

    @Slot(str)
    def _update_info_label(self, text):
        self.info_label.setText(text)

    @Slot()
    def _on_result_display_change(self):
        selected_dir = self.result_selector.currentData()
        if not self.results_data:
            return
            
        phase_map, data_map = None, {}
        if selected_dir == 'h' and 'h_unwrapped' in self.results_data and self.results_data['h_unwrapped']:
            phase_map = self.results_data['h_unwrapped'][0]
            data_map = {'h_phase': phase_map}
        elif selected_dir == 'v' and 'v_unwrapped' in self.results_data and self.results_data['v_unwrapped']:
            phase_map = self.results_data['v_unwrapped'][0]
            data_map = {'v_phase': phase_map}
            
        self.smoothed_tab.set_image(self._to_display(phase_map), data_map)

    def _on_complete(self, results):
        self.results_data = results
        self.process_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("处理完成！")
        
        h_res, v_res, c_res = None, None, None
        
        if 'h_unwrapped' in results and results['h_unwrapped']:
            h_res = results['h_unwrapped'][0]
        if 'v_unwrapped' in results and results['v_unwrapped']:
            v_res = results['v_unwrapped'][0]
        if 'combined' in results and results['combined']:
            # convert from RGB float to BGR uint8
            c_res_float = results['combined'][0]
            c_res = cv.cvtColor((c_res_float * 255).astype(np.uint8), cv.COLOR_RGB2BGR)

        # Display logic
        mode_id = self.mode_group.checkedId()
        self.result_selector.setVisible(mode_id == 2)

        if mode_id == 0: # Horizontal only
            self.smoothed_tab.set_image(self._to_display(h_res), {'h_phase': h_res})
            self.combined_tab.set_image(None)
        elif mode_id == 1: # Vertical only
            self.smoothed_tab.set_image(self._to_display(v_res), {'v_phase': v_res})
            self.combined_tab.set_image(None)
        else: # Both
            self.result_selector.setCurrentIndex(0)
            self.smoothed_tab.set_image(self._to_display(h_res), {'h_phase': h_res}) # Show horizontal by default
            self.combined_tab.set_image(c_res, {'h_phase': h_res, 'v_phase': v_res})
            
        self._update_info_label("点击上方图像以显示此处的光标处相位信息。")
        QMessageBox.information(self, "成功", "相位解包裹处理已成功完成。")
        
    def _to_display(self, phase_map):
        if phase_map is None: return None
        # Normalize to 0-255 and apply colormap
        norm = cv.normalize(phase_map, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        return cv.applyColorMap(norm, cv.COLORMAP_JET)

    def _show_reference_manual(self):
        """加载并显示参考手册对话框。"""
        # 构建手册文件的路径
        try:
            # 手册位于当前脚本的父目录中
            script_dir = os.path.dirname(os.path.abspath(__file__))
            manual_path = os.path.normpath(os.path.join(script_dir, '..', '格雷码解包裹操作手册.md'))
        except NameError:
            # 在某些无法获取 __file__ 的环境中提供备用路径
            manual_path = '../格雷码解包裹操作手册.md'

        if not os.path.exists(manual_path):
            QMessageBox.warning(self, "错误", f"未找到参考手册文件: '{manual_path}'")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("参考手册: 投影方向与解包裹方向")
        main_geo = self.geometry()
        dialog.setGeometry(main_geo.x() + 50, main_geo.y() + 50, 750, 600)
        
        layout = QVBoxLayout(dialog)
        
        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(True)

        try:
            with open(manual_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            text_browser.setMarkdown(markdown_content)
        except Exception as e:
            text_browser.setText(f"无法加载或解析手册文件。\n\n路径: {manual_path}\n错误: {e}")

        # 应用CSS样式以获得更好的可读性
        text_browser.setStyleSheet("""
            QTextBrowser {
                font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
                font-size: 15px;
                padding: 15px;
                background-color: #ffffff;
                border-radius: 4px;
                border: 1px solid #ddd;
            }
            h1, h2, h3 {
                font-weight: 600;
                margin-top: 20px;
                margin-bottom: 10px;
                padding-bottom: 5px;
                border-bottom: 1px solid #eaecef;
            }
            h1 { font-size: 24px; }
            h2 { font-size: 20px; }
            h3 { font-size: 16px; }
            p { line-height: 1.6; margin-bottom: 12px; }
            code {
                font-family: "Consolas", "Courier New", monospace;
                background-color: #f6f8fa;
                padding: 3px 5px;
                font-size: 14px;
                border-radius: 3px;
            }
            table {
                border-collapse: collapse;
                margin: 15px 0;
                font-size: 14px;
                width: 100%;
            }
            th, td {
                border: 1px solid #dfe2e5;
                padding: 8px 12px;
            }
            th {
                background-color: #f6f8fa;
                font-weight: bold;
            }
        """)

        layout.addWidget(text_browser)
        
        button_box = QHBoxLayout()
        button_box.addStretch()
        close_button = QPushButton("关闭")
        close_button.setDefault(True)
        close_button.clicked.connect(dialog.accept)
        button_box.addWidget(close_button)
        layout.addLayout(button_box)
        
        dialog.exec()

    def _on_error(self, message):
        self.process_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("处理失败！")
        QMessageBox.critical(self, "处理错误", message)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PhaseUnwrapperUI()
    window.show()
    sys.exit(app.exec()) 