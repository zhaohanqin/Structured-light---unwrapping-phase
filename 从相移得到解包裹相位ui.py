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

            # --- 水平解包裹流程 ---
            if mode in ['horizontal', 'both']:
                self.progress_update.emit(5, "开始水平解包裹流程...")
                images_h_unwrap, _ = self._load_split_fringes('h')
                v_gray_images, _ = self._load_gray_images(self.params['v_graycodes_path'])
                
                if images_h_unwrap and v_gray_images:
                    self.progress_update.emit(10, "[H] 计算包裹相位...")
                    wp_h = WrappedPhase(n=self.params['steps'])
                    wrapped_phase_h = wp_h.computeWrappedphase(images_h_unwrap)
                    wp_h.save_wrapped_phase(wrapped_phase_h, output_dir, "h_unwrap_from_v_fringe_", "vertical")

                    self.progress_update.emit(25, "[H] 解包裹相位...")
                    unwrapper_h = PhaseUnwrapper(n=self.params['gray_bits'], direction="horizontal")
                    unwrapped_phase_h = unwrapper_h.unwrap_phase(wrapped_phase_h, v_gray_images, save_results=True, show_results=False, basename="horizontal_unwrapped")
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
                    wrapped_phase_v = wp_v.computeWrappedphase(images_v_unwrap)
                    wp_v.save_wrapped_phase(wrapped_phase_v, output_dir, "v_unwrap_from_h_fringe_", "horizontal")

                    self.progress_update.emit(70, "[V] 解包裹相位...")
                    unwrapper_v = PhaseUnwrapper(n=self.params['gray_bits'], direction="vertical")
                    unwrapped_phase_v = unwrapper_v.unwrap_phase(wrapped_phase_v, h_gray_images, save_results=True, show_results=False, basename="vertical_unwrapped")
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
    # 当用户点击图像时，发射包含像素信息的信号
    pixel_info_signal = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.setText("无结果图像")
        self.pixmap = None
        self.raw_data = None  # 用于存储原始的、未处理的数据 (e.g., phase map)

    def set_image(self, cv_img, raw_data=None):
        self.raw_data = raw_data
        if cv_img is None:
            self.clear()
            self.setText("无结果图像")
            self.pixmap = None
            self.raw_data = None
            return

        h, w = cv_img.shape[:2]
        
        if len(cv_img.shape) == 2:
            # 灰度图像
            img_format = QImage.Format_Grayscale8
            bytes_per_line = w
            q_img = QImage(cv_img.data, w, h, bytes_per_line, img_format)
        else:
            # BGR彩色图像，转换为RGB
            cv_img_rgb = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
            img_format = QImage.Format_RGB888
            bytes_per_line = 3 * w
            q_img = QImage(cv_img_rgb.data, w, h, bytes_per_line, img_format)
        
        self.pixmap = QPixmap.fromImage(q_img)
        self.update_display()
        
    def update_display(self):
        if not self.pixmap: return
        self.setPixmap(self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self.update_display()

    def mousePressEvent(self, event):
        """处理鼠标点击事件，计算并发送像素信息。"""
        if self.raw_data is None or not self.pixmap:
            self.pixel_info_signal.emit("无图像数据")
            return

        label_size = self.size()
        pixmap_original_size = self.pixmap.size()

        if pixmap_original_size.width() == 0 or pixmap_original_size.height() == 0:
            return

        # 计算保持长宽比缩放后pixmap的实际大小和边距
        scaled_pixmap = self.pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.FastTransformation)
        x_offset = (label_size.width() - scaled_pixmap.width()) / 2
        y_offset = (label_size.height() - scaled_pixmap.height()) / 2
        
        click_pos = event.pos()
        pixmap_x = click_pos.x() - x_offset
        pixmap_y = click_pos.y() - y_offset

        # 检查点击是否在缩放后的pixmap内部
        if 0 <= pixmap_x < scaled_pixmap.width() and 0 <= pixmap_y < scaled_pixmap.height():
            # 将点击坐标从缩放后的pixmap空间转换到原始图像空间
            orig_h, orig_w = self.raw_data.shape[:2]
            img_x = int((pixmap_x / scaled_pixmap.width()) * orig_w)
            img_y = int((pixmap_y / scaled_pixmap.height()) * orig_h)
            
            if 0 <= img_x < orig_w and 0 <= img_y < orig_h:
                value = self.raw_data[img_y, img_x]
                if isinstance(value, np.ndarray):  # 彩色图像 (e.g., RGB)
                    info_str = f"坐标: ({img_x}, {img_y}),  值: {value}"
                else:  # 单通道图像 (e.g., phase)
                    info_str = f"坐标: ({img_x}, {img_y}),  相位值: {value:.4f}"
                self.pixel_info_signal.emit(info_str)
            else:
                self.pixel_info_signal.emit("就绪")
        else:
            self.pixel_info_signal.emit("就绪")


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
        self.status_bar.showMessage("就绪")

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
        help_button = QPushButton("📖 查看参考手册")
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
        layout.addWidget(path_group)

        # Algorithm Parameters
        param_group = QGroupBox("3. 算法参数")
        param_layout = QHBoxLayout(param_group)
        param_layout.addWidget(QLabel("相移步数:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(3, 20); self.steps_spin.setValue(4)
        param_layout.addWidget(self.steps_spin)
        param_layout.addWidget(QLabel("格雷码位数:"))
        self.gray_bits_spin = QSpinBox()
        self.gray_bits_spin.setRange(3, 10); self.gray_bits_spin.setValue(5)
        param_layout.addWidget(self.gray_bits_spin)
        param_layout.addStretch()
        layout.addWidget(param_group)
        
        # Output Path
        output_group = QGroupBox("4. 设置输出路径")
        output_layout = QVBoxLayout(output_group)
        self.output_path_edit = self._create_path_selector("输出根目录:", output_layout, "reconstruction_results")
        layout.addWidget(output_group)

        # Action Button
        self.process_button = QPushButton("🚀 开始重建")
        self.process_button.setFixedHeight(45)
        self.process_button.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #c8e6c9;")
        self.process_button.clicked.connect(self._start_reconstruction)
        layout.addWidget(self.process_button)

        layout.addStretch()
        self._on_mode_change()
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        return scroll

    def _create_path_selector(self, label_text, parent_layout, default_path=""):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(label_text))
        edit = QLineEdit(default_path)
        button = QPushButton("浏览...")
        button.clicked.connect(lambda: self._browse_folder(edit))
        layout.addWidget(edit)
        layout.addWidget(button)
        parent_layout.addWidget(row)
        return edit

    def _create_results_panel(self):
        panel = QTabWidget()
        self.h_unwrapped_tab = InteractiveImage()
        self.v_unwrapped_tab = InteractiveImage()
        self.combined_tab = InteractiveImage()

        # 连接信号到状态栏更新槽函数
        self.h_unwrapped_tab.pixel_info_signal.connect(self._update_status_bar)
        self.v_unwrapped_tab.pixel_info_signal.connect(self._update_status_bar)
        self.combined_tab.pixel_info_signal.connect(self._update_status_bar)

        panel.addTab(self.h_unwrapped_tab, "水平解包裹结果")
        panel.addTab(self.v_unwrapped_tab, "垂直解包裹结果")
        panel.addTab(self.combined_tab, "组合相位图")
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

    def _show_reference_manual(self):
        """加载并显示参考手册对话框。"""
        # 构建手册文件的路径
        try:
            # 脚本和手册在同一个目录下
            script_dir = os.path.dirname(os.path.abspath(__file__))
            manual_path = os.path.join(script_dir, '格雷码解包裹操作手册.md')
        except NameError:
            # 在某些无法获取 __file__ 的环境中提供备用路径
            manual_path = '格雷码解包裹操作手册.md'

        if not os.path.exists(manual_path):
            QMessageBox.warning(self, "错误", f"未找到参考手册文件: '{manual_path}'")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("参考手册: 投影方向与解包裹方向")
        main_geo = self.geometry()
        dialog.setGeometry(main_geo.x() + 60, main_geo.y() + 60, 750, 600)
        
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

        try: # Validate inputs before starting thread
            if not params['fringes_path'] or not params['output']: raise ValueError("请指定相移图像文件夹和输出目录。")
            if params['mode'] in ['horizontal', 'both'] and not params['v_graycodes_path']: raise ValueError("水平模式需要垂直格雷码文件夹。")
            if params['mode'] in ['vertical', 'both'] and not params['h_graycodes_path']: raise ValueError("垂直模式需要水平格雷码文件夹。")
        except ValueError as e:
            QMessageBox.warning(self, "输入错误", str(e))
            return
            
        self.process_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage("准备开始处理...")

        self.thread = ReconstructionThread(params)
        self.thread.progress_update.connect(lambda v, m: self.status_bar.showMessage(f"进度: {v}% - {m}"))
        self.thread.processing_complete.connect(self._on_complete)
        self.thread.processing_error.connect(self._on_error)
        self.thread.start()

    def _on_complete(self, results):
        self.status_bar.showMessage("处理完成！结果已显示。")
        self.process_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if 'h_unwrapped' in results:
            raw_data = results['h_unwrapped']
            display_img = self._to_display_color(raw_data)
            self.h_unwrapped_tab.set_image(display_img, raw_data)
        if 'v_unwrapped' in results:
            raw_data = results['v_unwrapped']
            display_img = self._to_display_color(raw_data)
            self.v_unwrapped_tab.set_image(display_img, raw_data)
        if 'combined' in results:
            # Convert RGB float to BGR uint8 for display
            rgb_float = results['combined']
            bgr_uint8 = cv.cvtColor((rgb_float * 255).astype(np.uint8), cv.COLOR_RGB2BGR)
            self.combined_tab.set_image(bgr_uint8, raw_data=rgb_float)

        QMessageBox.information(self, "成功", "重建流程已成功完成。")

    def _on_error(self, message):
        self.status_bar.showMessage("处理失败！")
        self.process_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "处理错误", f"发生错误: {message}")

    @Slot(str)
    def _update_status_bar(self, text):
        """更新状态栏的文本。"""
        self.status_bar.showMessage(text)

    def _to_display_color(self, phase_map):
        if phase_map is None: return None
        norm = cv.normalize(phase_map, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        return cv.applyColorMap(norm, cv.COLORMAP_JET)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Reconstruct3D_UI()
    window.show()
    sys.exit(app.exec()) 