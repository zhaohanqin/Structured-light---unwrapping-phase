import sys
import os
import numpy as np
import cv2 as cv
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QSpinBox, QLineEdit, 
                               QPushButton, QFileDialog, QRadioButton, QButtonGroup, 
                               QGroupBox, QTabWidget, QSplitter, QMessageBox, 
                               QProgressBar, QScrollArea, QDialog, QTextBrowser)
from PySide6.QtGui import QPixmap, QImage, QColor, QPalette, QPainter, QPen
from PySide6.QtCore import Qt, Signal, Slot, QThread

# 导入核心处理逻辑
from wrapped_phase import WrappedPhase
from phase_unwrapper import PhaseUnwrapper, generate_combined_phase_image

class ReconstructionThread(QThread):
    """用于后台执行完整重建流程的线程"""
    progress_update = Signal(int, str)
    processing_complete = Signal(dict)
    processing_error = Signal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            results = {}
            mode = self.params['mode']
            output_dir = self.params['output']

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 加载全黑和全白图像（如果有）
            black_image, white_image = self._load_black_white_images()

            # --- 水平解包裹流程 ---
            if mode in ['horizontal', 'both']:
                self.progress_update.emit(5, "开始水平解包裹流程...")
                images_h_unwrap, _ = self._load_split_fringes('h')
                v_gray_images, _ = self._load_gray_images(self.params['v_graycodes_path'])
                
                if images_h_unwrap and v_gray_images:
                    self.progress_update.emit(10, "[H] 计算包裹相位...")
                    wp_h = WrappedPhase(n=self.params['steps'])
                    wrapped_phase_h = wp_h.computeWrappedphase(images_h_unwrap, black_image, white_image)
                    wp_h.save_wrapped_phase(wrapped_phase_h, output_dir, "h_unwrap_from_v_fringe_", "vertical")

                    self.progress_update.emit(25, "[H] 解包裹相位...")
                    unwrapper_h = PhaseUnwrapper(n=self.params['gray_bits'], direction="horizontal")
                    unwrapped_phase_h = unwrapper_h.unwrap_phase(wrapped_phase_h, v_gray_images, 
                                                                black_image=black_image, white_image=white_image,
                                                                save_results=True, show_results=False, 
                                                                basename="horizontal_unwrapped")
                    results['h_unwrapped'] = unwrapped_phase_h
                else:
                    raise ValueError("水平解包裹所需图像不完整。")
            
            # --- 垂直解包裹流程 ---
            if mode in ['vertical', 'both']:
                self.progress_update.emit(50, "开始垂直解包裹流程...")
                _, images_v_unwrap = self._load_split_fringes('v')
                h_gray_images, _ = self._load_gray_images(self.params['h_graycodes_path'])

                if images_v_unwrap and h_gray_images:
                    self.progress_update.emit(55, "[V] 计算包裹相位...")
                    wp_v = WrappedPhase(n=self.params['steps'])
                    wrapped_phase_v = wp_v.computeWrappedphase(images_v_unwrap, black_image, white_image)
                    wp_v.save_wrapped_phase(wrapped_phase_v, output_dir, "v_unwrap_from_h_fringe_", "horizontal")

                    self.progress_update.emit(70, "[V] 解包裹相位...")
                    unwrapper_v = PhaseUnwrapper(n=self.params['gray_bits'], direction="vertical")
                    unwrapped_phase_v = unwrapper_v.unwrap_phase(wrapped_phase_v, h_gray_images, 
                                                                black_image=black_image, white_image=white_image,
                                                                save_results=True, show_results=False, 
                                                                basename="vertical_unwrapped")
                    results['v_unwrapped'] = unwrapped_phase_v
                else:
                    raise ValueError("垂直解包裹所需图像不完整。")
            
            # --- 合并结果 ---
            if mode == 'both' and 'h_unwrapped' in results and 'v_unwrapped' in results:
                self.progress_update.emit(95, "合并最终结果...")
                output_path = os.path.join(output_dir, "final_combined_phase.png")
                results['combined'] = generate_combined_phase_image(results['h_unwrapped'], results['v_unwrapped'], output_path)

            self.progress_update.emit(100, "全部处理完成!")
            self.processing_complete.emit(results)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.processing_error.emit(str(e))

    def _load_black_white_images(self):
        """加载全黑全白图像"""
        black_image = None
        white_image = None

        # 加载全黑图像
        if 'black_image_path' in self.params and self.params['black_image_path']:
            try:
                black_path = self.params['black_image_path']
                black_image = cv.imread(black_path, -1)
                if black_image is not None and len(black_image.shape) > 2:
                    black_image = cv.cvtColor(black_image, cv.COLOR_BGR2GRAY)
                self.progress_update.emit(2, f"加载全黑图像: {os.path.basename(black_path)}")
            except Exception as e:
                self.progress_update.emit(2, f"加载全黑图像失败: {str(e)}")
        
        # 加载全白图像
        if 'white_image_path' in self.params and self.params['white_image_path']:
            try:
                white_path = self.params['white_image_path']
                white_image = cv.imread(white_path, -1)
                if white_image is not None and len(white_image.shape) > 2:
                    white_image = cv.cvtColor(white_image, cv.COLOR_BGR2GRAY)
                self.progress_update.emit(4, f"加载全白图像: {os.path.basename(white_path)}")
            except Exception as e:
                self.progress_update.emit(4, f"加载全白图像失败: {str(e)}")
        
        return black_image, white_image

    def _load_images_from_folder(self, folder_path, expected_count=None):
        if not os.path.isdir(folder_path): return [], []
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        image_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(tuple(image_extensions))])
        
        if expected_count and len(image_paths) > expected_count:
            image_paths = image_paths[:expected_count]

        images = [cv.imread(p, -1) for p in image_paths]
        basenames = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
        return [img for img in images if img is not None], basenames
    
    def _load_split_fringes(self, part='h'):
        folder_path = self.params['fringes_path']
        steps = self.params['steps']
        all_imgs, _ = self._load_images_from_folder(folder_path, expected_count=2*steps)
        if len(all_imgs) < 2*steps:
            return None, None
        
        images_for_h = all_imgs[0:steps]
        images_for_v = all_imgs[steps:2*steps]
        
        if part == 'h': return images_for_h, None
        if part == 'v': return None, images_for_v
        return images_for_h, images_for_v

    def _load_gray_images(self, folder_path):
        return self._load_images_from_folder(folder_path, expected_count=self.params['gray_bits'])

class InteractiveImage(QLabel):
    """可交互的图像显示控件，用于显示相位值"""
    # 当鼠标悬停时，发送像素信息 (x, y, value)
    pixel_info_signal = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.setText("无结果图像")
        self.pixmap = None
        self.raw_data = None
        self.h_phase_data = None
        self.v_phase_data = None
        self.setMouseTracking(True) # 开启鼠标跟踪以接收悬停事件
        self.hover_position = None

    def set_image(self, cv_img, raw_data=None, h_phase_data=None, v_phase_data=None):
        """设置显示的图像和用于查询的原始数据。"""
        if cv_img is None:
            self.clear()
            self.setText("无结果图像")
            self.pixmap = None
            self.raw_data = None
            self.h_phase_data = None
            self.v_phase_data = None
            return

        self.raw_data = raw_data
        self.h_phase_data = h_phase_data
        self.v_phase_data = v_phase_data
        h, w = cv_img.shape[:2]
        if len(cv_img.shape) == 2:
            img_format = QImage.Format_Grayscale8
            bytes_per_line = w
        else:
            cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
            img_format = QImage.Format_RGB888
            bytes_per_line = 3 * w
        
        q_img = QImage(cv_img.data, w, h, bytes_per_line, img_format)
        self.pixmap = QPixmap.fromImage(q_img)
        self.update_display()
        
    def update_display(self):
        if not self.pixmap: return
        self.setPixmap(self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self.update_display()

    def mouseMoveEvent(self, event):
        """处理鼠标悬停事件，计算并发送像素信息。"""
        if self.pixmap is None:
            self.hover_position = None
            self.update()
            self.pixel_info_signal.emit("")
            return

        self.hover_position = event.pos()
        self.update()

        if self.raw_data is None:
            return

        widget_size = self.size()
        pixmap_size = self.pixmap.size()
        
        scaled_pixmap = self.pixmap.scaled(widget_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scaled_pixmap_size = scaled_pixmap.size()

        # 计算图像在QLabel中的偏移量
        offset_x = (widget_size.width() - scaled_pixmap_size.width()) / 2
        offset_y = (widget_size.height() - scaled_pixmap_size.height()) / 2

        pos_in_label = event.pos()
        x_in_pixmap = pos_in_label.x() - offset_x
        y_in_pixmap = pos_in_label.y() - offset_y

        # 检查鼠标是否在缩放后的图像范围内
        if 0 <= x_in_pixmap < scaled_pixmap_size.width() and 0 <= y_in_pixmap < scaled_pixmap_size.height():
            original_width = self.raw_data.shape[1]
            original_height = self.raw_data.shape[0]

            # 将鼠标坐标从缩放后的图像映射回原始图像
            img_x = int(x_in_pixmap * original_width / scaled_pixmap_size.width())
            img_y = int(y_in_pixmap * original_height / scaled_pixmap_size.height())
            
            # 确保坐标在有效范围内
            img_x = max(0, min(img_x, original_width - 1))
            img_y = max(0, min(img_y, original_height - 1))

            info_text = ""
            # 组合视图，拥有独立的 H/V 相位数据
            if self.h_phase_data is not None and self.v_phase_data is not None:
                info_parts = [f"坐标: ({img_x}, {img_y})"]
                
                # 确保坐标有效
                if (img_y < self.h_phase_data.shape[0] and img_x < self.h_phase_data.shape[1] and
                    img_y < self.v_phase_data.shape[0] and img_x < self.v_phase_data.shape[1]):
                    
                    h_phase = self.h_phase_data[img_y, img_x]
                    h_period = h_phase / (2 * np.pi)
                    info_parts.append(f"水平相位: {h_phase:.4f} rad, 周期: {h_period:.4f}")

                    v_phase = self.v_phase_data[img_y, img_x]
                    v_period = v_phase / (2 * np.pi)
                    info_parts.append(f"垂直相位: {v_phase:.4f} rad, 周期: {v_period:.4f}")
                
                info_text = " | ".join(info_parts)
            
            # 单通道视图（如 H 或 V 结果）
            elif len(self.raw_data.shape) == 2:
                value = self.raw_data[img_y, img_x]
                period = value / (2 * np.pi)
                info_text = f"坐标: ({img_x}, {img_y}) | 相位值: {value:.4f} rad | 周期数: {period:.4f}"

            # 其他多通道图像 (如原始的组合图)
            else:
                value = self.raw_data[img_y, img_x]
                val_str = ", ".join([f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in value])
                info_text = f"坐标: ({img_x}, {img_y}) | 值: ({val_str})"
            
            self.pixel_info_signal.emit(info_text)
        else:
            self.pixel_info_signal.emit("") # 鼠标在图像外，发送空字符串

    def leaveEvent(self, event):
        """当鼠标离开控件时，清空信息。"""
        self.hover_position = None
        self.update()
        self.pixel_info_signal.emit("")
        super().leaveEvent(event)

    def paintEvent(self, event):
        """重写绘制事件以添加交互式覆盖。"""
        super().paintEvent(event)

        if self.hover_position and self.pixmap:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            x = self.hover_position.x()
            y = self.hover_position.y()

            # 绘制十字准线
            painter.setPen(QPen(QColor(0, 255, 255, 150), 1, Qt.DashLine))
            painter.drawLine(x, 0, x, self.height())
            painter.drawLine(0, y, self.width(), y)
            
            painter.end()


class Reconstruct3D_UI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("全流程3D重建工具")
        self.setGeometry(100, 100, 1200, 700)
        self._set_style()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 修正初始化顺序：先创建结果面板，再创建控制面板
        self.results_panel = self._create_results_panel()
        control_panel = self._create_control_panel()
        
        splitter.addWidget(control_panel)
        splitter.addWidget(self.results_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_message = "就绪"
        self.status_bar.showMessage(self.status_message)

    def _set_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f5; }
            QGroupBox {
                font-weight: bold; border: 1px solid #c0c0c0;
                border-radius: 5px; margin-top: 1ex; padding-top: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; subcontrol-position: top left;
                padding: 0 5px; left: 10px;
            }
            QPushButton {
                background-color: #e1e1e1; border: 1px solid #adadad;
                padding: 6px 12px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #d1d1d1; }
            QLineEdit, QSpinBox { padding: 4px; border: 1px solid #c0c0c0; border-radius: 4px; }
            QTabWidget::pane { border-top: 1px solid #c0c0c0; }
            QTabBar::tab { padding: 8px 15px; border: 1px solid #c0c0c0; border-bottom: none; 
                           border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #f5f5f5; }
            QTabBar::tab:!selected { background: #e1e1e1; margin-top: 2px;}
        """)

    def _create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Help button
        help_button_layout = QHBoxLayout()
        help_button_layout.addStretch()
        help_button = QPushButton("查看参考手册")
        help_button.setStyleSheet("padding: 4px 8px; font-size: 13px;") # Make it less prominent
        help_button.clicked.connect(self._show_reference_manual)
        help_button_layout.addWidget(help_button)
        layout.addLayout(help_button_layout)

        # Mode Selection
        mode_group = QGroupBox("1. 选择处理模式")
        mode_layout = QHBoxLayout(mode_group)
        self.mode_group = QButtonGroup(self)
        self.h_radio = QRadioButton("仅水平解包裹")
        self.v_radio = QRadioButton("仅垂直解包裹")
        self.both_radio = QRadioButton("双向合并")

        self.h_radio.setToolTip("计算水平方向的相位。\n需要垂直方向的相移条纹和垂直方向的格雷码图像。")
        self.v_radio.setToolTip("计算垂直方向的相位。\n需要水平方向的相移条纹和水平方向的格雷码图像。")
        self.both_radio.setToolTip("合并水平和垂直两个方向的解包裹结果，生成最终的相位图。\n这是最常用的模式，可以提供最完整的三维表面。")
        
        self.mode_group.addButton(self.h_radio, 0)
        self.mode_group.addButton(self.v_radio, 1)
        self.mode_group.addButton(self.both_radio, 2)
        self.both_radio.setChecked(True)
        self.mode_group.buttonClicked.connect(self._on_mode_change)
        mode_layout.addWidget(self.h_radio)
        mode_layout.addWidget(self.v_radio)
        mode_layout.addWidget(self.both_radio)
        layout.addWidget(mode_group)

        # Path Inputs
        path_group = QGroupBox("2. 设置输入路径")
        path_layout = QVBoxLayout(path_group)
        self.fringes_path_edit = self._create_path_selector("相移图像 (文件夹):", path_layout)
        self.v_gray_path_edit = self._create_path_selector("垂直格雷码 (解水平包裹相位):", path_layout)
        self.h_gray_path_edit = self._create_path_selector("水平格雷码 (解垂直包裹相位):", path_layout)

        self.fringes_path_edit.setToolTip("包含所有相移条纹图的文件夹。\n程序会根据步数N自动区分用于水平解包裹的垂直条纹(前N张)\n和用于垂直解包裹的水平条纹(后N张)。")
        self.v_gray_path_edit.setToolTip("包含垂直格雷码图像的文件夹。\n这些图像用于辅助水平方向的相位解包裹。")
        self.h_gray_path_edit.setToolTip("包含水平格雷码图像的文件夹。\n这些图像用于辅助垂直方向的相位解包裹。")
        
        # 添加全黑全白图像输入路径
        black_white_group = QGroupBox("3. 全黑/全白图像 (可选)")
        black_white_group.setToolTip("全黑图像用于环境光校正，全白图像用于反射率校正。\n这些图像可以提高解包裹相位的质量和鲁棒性。")
        black_white_layout = QVBoxLayout(black_white_group)
        self.black_image_path_edit = self._create_path_selector("全黑图像:", black_white_layout, allow_clear=True)
        self.white_image_path_edit = self._create_path_selector("全白图像:", black_white_layout, allow_clear=True)
        path_layout.addWidget(black_white_group)

        layout.addWidget(path_group)

        # Algorithm Parameters
        param_group = QGroupBox("4. 算法参数")
        param_layout = QHBoxLayout(param_group)
        param_layout.addWidget(QLabel("相移步数:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(3, 20); self.steps_spin.setValue(4)
        self.steps_spin.setToolTip("设置相移的步数 (N)。\n此值必须与生成相移条纹时使用的步数一致。\n常用值为3, 4, 8。")
        param_layout.addWidget(self.steps_spin)
        param_layout.addWidget(QLabel("格雷码位数:"))
        self.gray_bits_spin = QSpinBox()
        self.gray_bits_spin.setRange(3, 10); self.gray_bits_spin.setValue(5)
        self.gray_bits_spin.setToolTip("设置格雷码的位数。\n此值必须与生成格雷码图像时使用的位数一致。\n位数越多，能解包裹的条纹周期数也越多。")
        param_layout.addWidget(self.gray_bits_spin)
        param_layout.addStretch()
        layout.addWidget(param_group)
        
        # Output Path
        output_group = QGroupBox("5. 设置输出路径")
        output_layout = QVBoxLayout(output_group)
        self.output_path_edit = self._create_path_selector("输出根目录:", output_layout, "reconstruction_results")
        self.output_path_edit.setToolTip("所有中间和最终结果都将保存在此文件夹中。")
        layout.addWidget(output_group)

        # Action Button
        self.process_button = QPushButton("开始重建")
        self.process_button.setFixedHeight(45)
        self.process_button.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #c8e6c9;")
        self.process_button.setToolTip("开始执行解包裹和重建流程。")
        self.process_button.clicked.connect(self._start_reconstruction)
        layout.addWidget(self.process_button)

        layout.addStretch()
        self._on_mode_change()
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        return scroll

    def _create_path_selector(self, label_text, parent_layout, default_path="", allow_clear=False):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(label_text))
        edit = QLineEdit(default_path)
        button = QPushButton("浏览...")
        button.clicked.connect(lambda: self._browse_folder(edit))
        layout.addWidget(edit)
        layout.addWidget(button)
        
        if allow_clear:
            clear_button = QPushButton("清除")
            clear_button.clicked.connect(lambda: edit.setText(""))
            layout.addWidget(clear_button)
            
        parent_layout.addWidget(row)
        return edit

    def _create_results_panel(self):
        panel = QTabWidget()
        self.h_unwrapped_tab = InteractiveImage()
        self.v_unwrapped_tab = InteractiveImage()
        self.combined_tab = InteractiveImage()
        panel.addTab(self.h_unwrapped_tab, "水平解包裹结果")
        panel.addTab(self.v_unwrapped_tab, "垂直解包裹结果")
        panel.addTab(self.combined_tab, "组合相位图")

        # 连接信号到状态栏更新槽
        self.h_unwrapped_tab.pixel_info_signal.connect(self._update_status_bar)
        self.v_unwrapped_tab.pixel_info_signal.connect(self._update_status_bar)
        self.combined_tab.pixel_info_signal.connect(self._update_status_bar)

        return panel

    def _browse_folder(self, line_edit):
        path = QFileDialog.getExistingDirectory(self, "选择文件夹", line_edit.text())
        if path:
            line_edit.setText(path)
            
    def _on_mode_change(self):
        mode = self.mode_group.checkedId()
        self.fringes_path_edit.parentWidget().setVisible(True)
        self.v_gray_path_edit.parentWidget().setVisible(mode in [0, 2])
        self.h_gray_path_edit.parentWidget().setVisible(mode in [1, 2])
        self.results_panel.setTabVisible(0, mode in [0, 2])
        self.results_panel.setTabVisible(1, mode in [1, 2])
        self.results_panel.setTabVisible(2, mode == 2)

    @Slot(int, str)
    def _update_progress(self, value, message):
        """更新进度条和状态栏消息。"""
        self.progress_bar.setValue(value)
        self.status_bar.showMessage(f"进度: {value}% - {message}")

    @Slot(str)
    def _update_status_bar(self, message):
        """更新状态栏以显示像素信息或恢复默认消息。"""
        # 只在空闲时更新像素信息，避免覆盖处理进度
        if not self.process_button.isEnabled():
            return
        
        if message:
            self.status_bar.showMessage(message)
        else:
            self.status_bar.showMessage(self.status_message)

    def _show_reference_manual(self):
        """加载并显示参考手册对话框。"""
        # 将手册内容直接嵌入代码中，避免文件依赖
        manual_content = """
# 结构光三维重建参考手册

欢迎使用本套结构光三维重建工具。本手册旨在阐明一个核心且容易混淆的概念：**投影条纹的方向**与**解包裹相位方向**之间的关系。

---

## 核心问题：为什么条纹方向和解包裹方向是垂直的？

简单来说：**我们用"垂直的尺子"去量"水平的长度"，用"水平的尺子"去量"垂直的高度"。**

在结构光技术中，投影的条纹就是我们用来测量物体形状的"数字尺子"。

### 1. 解水平包裹相位（测量物体水平方向的形状）

*   **目标**：我们想知道物体表面从左到右（即沿 **X轴**）的形状轮廓是如何变化的。
*   **方法**：为了测量水平方向上的变化，我们投射的"尺子"必须在水平方向上具有刻度。什么样的条纹能在水平方向上形成刻度呢？答案是**垂直条纹**。
*   **类比**：
    *   想象一道**栅栏**，栅栏的木条是**垂直**的。
    *   当您**水平**地沿着栅栏走，您会依次经过"木条-空隙-木条-空隙..."，这种变化为您提供了水平方向上的位置信息。
    *   同理，当**垂直的条纹光**投射到物体上时，正是这些条纹的左右扭曲，才揭示了物体在**水平方向**上的深度信息。

**结论**：因此，我们使用 **垂直条纹 (Vertical Fringes)** 来进行 **水平方向的解包裹 (Horizontal Unwrapping)**。

> **对应文件**:
>
> *   相移图像: `I1.png` 到 `IN.png`
> *   格雷码图像: 包含**垂直条纹**的格雷码图

---

### 2. 解垂直包裹相位（测量物体垂直方向的形状）

*   **目标**：我们想知道物体表面从上到下（即沿 **Y轴**）的形状轮廓是如何变化的。
*   **方法**：与上面同理，我们需要一把能在垂直方向上提供刻度的"尺子"。这个"尺子"就是**水平条纹**。
*   **类比**：
    *   想象一个**梯子**，梯子的横杆是**水平**的。
    *   当您**垂直**地向上爬梯子时，您会依次经过"横杆-空隙-横杆-空隙..."，这种变化为您提供了垂直方向上的位置信息。
    *   同理，当**水平的条纹光**投射到物体上时，正是这些条纹的上下扭曲，才揭示了物体在**垂直方向**上的深度信息。

**结论**：因此，我们使用 **水平条纹 (Horizontal Fringes)** 来进行 **垂直方向的解包裹 (Vertical Unwrapping)**。

> **对应文件**:
>
> *   相移图像: `I(N+1).png` 到 `I(2N).png`
> *   格雷码图像: 包含**水平条纹**的格雷码图

---

## 快速参考表

| 您想做什么？                 | 您需要投影什么方向的条纹？ | 对应哪组相移图像？     | 对应哪种格雷码？   |
| :--------------------------- | :--------------------------- | :--------------------- | :----------------- |
| **解水平包裹相位**           | ✅ **垂直**条纹              | `I1` 到 `IN`           | **垂直**格雷码     |
| **解垂直包裹相位**           | ✅ **水平**条纹              | `I(N+1)` 到 `I(2N)` | **水平**格雷码     |

## 关于格雷码

格雷码的作用是为相移法计算出的包裹相位提供一个绝对的"基准线"，以解决相位模糊的问题。因此，格雷码图案的方向必须与其辅助的相移条纹方向**保持一致**。

*   用**垂直**的格雷码图案，来辅助**垂直**的相移条纹，共同完成**水平**方向的解包裹。
*   用**水平**的格雷码图案，来辅助**水平**的相移条纹，共同完成**垂直**方向的解包裹。

希望这份手册能帮助您更好地理解和使用本工具。
"""
        
        dialog = QDialog(self)
        dialog.setWindowTitle("参考手册: 投影方向与解包裹方向")
        main_geo = self.geometry()
        dialog.setGeometry(main_geo.x() + 60, main_geo.y() + 60, 750, 600)
        
        layout = QVBoxLayout(dialog)
        
        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(True)
        text_browser.setMarkdown(manual_content)

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

    def _start_reconstruction(self):
        params = {
            'mode': ['horizontal', 'vertical', 'both'][self.mode_group.checkedId()],
            'fringes_path': self.fringes_path_edit.text(),
            'v_graycodes_path': self.v_gray_path_edit.text(),
            'h_graycodes_path': self.h_gray_path_edit.text(),
            'steps': self.steps_spin.value(),
            'gray_bits': self.gray_bits_spin.value(),
            'output': self.output_path_edit.text()
        }

        # 添加全黑全白图像路径
        black_path = self.black_image_path_edit.text()
        if black_path and os.path.isfile(black_path):
            params['black_image_path'] = black_path
        
        white_path = self.white_image_path_edit.text()
        if white_path and os.path.isfile(white_path):
            params['white_image_path'] = white_path

        try: # Validate inputs before starting thread
            if not params['fringes_path'] or not params['output']: raise ValueError("请指定相移图像文件夹和输出目录。")
            if params['mode'] in ['horizontal', 'both'] and not params['v_graycodes_path']: raise ValueError("水平模式需要垂直格雷码文件夹。")
            if params['mode'] in ['vertical', 'both'] and not params['h_graycodes_path']: raise ValueError("垂直模式需要水平格雷码文件夹。")
        except ValueError as e:
            QMessageBox.warning(self, "输入错误", str(e))
            return
            
        self.process_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_message = "准备开始处理..."
        self.status_bar.showMessage(self.status_message)

        self.thread = ReconstructionThread(params)
        self.thread.progress_update.connect(self._update_progress)
        self.thread.processing_complete.connect(self._on_complete)
        self.thread.processing_error.connect(self._on_error)
        self.thread.start()

    def _on_complete(self, results):
        self.status_message = "处理完成！结果已显示。"
        self.status_bar.showMessage(self.status_message)
        self.process_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if 'h_unwrapped' in results:
            display_img_h = self._to_display_color(results['h_unwrapped'])
            self.h_unwrapped_tab.set_image(display_img_h, raw_data=results['h_unwrapped'])
        if 'v_unwrapped' in results:
            display_img_v = self._to_display_color(results['v_unwrapped'])
            self.v_unwrapped_tab.set_image(display_img_v, raw_data=results['v_unwrapped'])
        if 'combined' in results:
            # Convert RGB float to BGR uint8 for display
            rgb_float = results['combined']
            bgr_uint8 = cv.cvtColor((rgb_float * 255).astype(np.uint8), cv.COLOR_RGB2BGR)
            self.combined_tab.set_image(
                bgr_uint8, 
                raw_data=rgb_float, 
                h_phase_data=results.get('h_unwrapped'),
                v_phase_data=results.get('v_unwrapped')
            )

        QMessageBox.information(self, "成功", "重建流程已成功完成。")

    def _on_error(self, message):
        self.status_message = "处理失败！"
        self.status_bar.showMessage(self.status_message)
        self.process_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "处理错误", f"发生错误: {message}")

    def _to_display_color(self, phase_map):
        if phase_map is None: return None
        norm = cv.normalize(phase_map, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        return cv.applyColorMap(norm, cv.COLORMAP_JET)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Reconstruct3D_UI()
    window.show()
    sys.exit(app.exec()) 