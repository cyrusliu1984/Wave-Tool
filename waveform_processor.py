import sys
import os
import json
import pandas as pd
import numpy as np
import peakutils
from scipy.signal import savgol_filter
import chardet
import matplotlib
matplotlib.use('Qt5Agg')  # 强制使用Qt5后端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QFileDialog, QLabel, QVBoxLayout, QHBoxLayout, 
                             QWidget, QTextEdit, QProgressBar, QMessageBox,
                             QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from io import BytesIO  # 内存缓冲区
from PIL import Image

# -------------------------- 模型相关导入与配置 --------------------------
import torch
from torchvision import transforms
try:
    from model import swin_tiny_patch4_window7_224 as create_model
except ImportError:
    raise ImportError("请确保model.py在当前目录，且包含swin_tiny_patch4_window7_224函数")


def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        # 打包后：_MEIPASS是PyInstaller的临时解压目录
        return os.path.join(sys._MEIPASS, relative_path)
    # 未打包时：使用当前文件夹路径
    return os.path.join(os.path.abspath("."), relative_path)

# 模型配置（用户需根据实际路径修改）
# MODEL_WEIGHT_PATH = "./model-2.pth"  # 模型权重路径（.pth格式）
# CLASS_JSON_PATH = "./class_indices.json"  # 类别映射文件
MODEL_WEIGHT_PATH = get_resource_path("model.pth")
CLASS_JSON_PATH = get_resource_path("class_indices.json")
IMG_SIZE = 224





# 图像预处理（与模型训练时保持一致）
def get_data_transform():
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.43)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# 初始化模型（全局初始化，避免重复加载）
def init_waveform_model(device):
    try:
        # 加载类别映射
        assert os.path.exists(CLASS_JSON_PATH), f"类别文件不存在：{CLASS_JSON_PATH}"
        with open(CLASS_JSON_PATH, "r", encoding="utf-8") as f:
            class_indict = json.load(f)
        
        # 创建模型（num_classes与训练时一致）
        model = create_model(num_classes=2).to(device)
        
        # 加载模型权重
        assert os.path.exists(MODEL_WEIGHT_PATH), f"模型权重不存在：{MODEL_WEIGHT_PATH}"
        model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location=device))
        
        # 设为评估模式
        model.eval()
        return model, class_indict
    except Exception as e:
        raise RuntimeError(f"模型初始化失败：{str(e)}")

# -------------------------- 基础组件定义 --------------------------
class MplCanvas(FigureCanvas):
    """matplotlib画布类，用于显示波形"""
    def __init__(self, parent=None, width=12, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)  # 确保正确继承FigureCanvasQTAgg
        self.setParent(parent)
        self.fig.tight_layout()  # 自动调整布局

# -------------------------- 数据处理线程 --------------------------
class ProcessingThread(QThread):
    update_progress = pyqtSignal(int)
    update_status = pyqtSignal(str)
    processing_finished = pyqtSignal(list, dict)  # (处理文件列表, 日志数据字典)

    def __init__(self, file_path, threshold):
        super().__init__()
        self.file_path = file_path
        self.threshold = threshold
        self.processed_files = []
        self.log_data = {}  # {文件名: (maxnum, max_num_09, max_num_01, tr, tf, tw)}

    def run(self):
        try:
            self.process_file(self.file_path, self.threshold)
            self.update_status.emit("Processing completed!")
            self.update_progress.emit(100)
            self.processing_finished.emit(self.processed_files, self.log_data)
        except Exception as e:
            self.update_status.emit(f"Processing error: {str(e)}")

    def process_file(self, file_path, threshold):
        self.update_status.emit(f"Starting processing: {os.path.basename(file_path)}")
        self.update_progress.emit(10)
        
        weith = 400
        base_file_name = os.path.basename(file_path)
        log_file_path = os.path.join(os.path.dirname(file_path), "log.csv")
        
        try:
            # 1. 读取CSV
            with open(file_path, 'rb') as f:
                encoding = chardet.detect(f.read())['encoding'] or 'utf-8'
            df = pd.read_csv(file_path, low_memory=False, encoding=encoding)
            self.update_progress.emit(20)

            # 2. 数据预处理
            df = df.iloc[1:,:].copy()
            col = list(df.columns)
            df[col] = df[col].apply(pd.to_numeric, errors='coerce').fillna(0.0)
            df = pd.DataFrame(df, dtype='float')
            self.update_progress.emit(30)

            # 3. 检测必要列
            time_col = next((c for c in col if 'time' in str(c).lower() or '时间' in str(c)), None)
            channel_a_col = next((c for c in col if 'a' in str(c).lower() or '通道a' in str(c).lower()), None)
            if not time_col or not channel_a_col:
                raise ValueError("CSV must contain 'time' and 'channel A' columns")
            self.update_progress.emit(35)

            # 4. 尖峰检测
            self.update_status.emit("Detecting peaks...")
            indexes = peakutils.indexes(df[channel_a_col], thres=threshold, min_dist=weith, thres_abs=True)
            self.update_progress.emit(40)
            if len(indexes) == 0:
                self.update_status.emit("No peaks detected")
                return

            # 5. 生成结果
            result_dir = os.path.join(os.path.dirname(file_path), "result")
            os.makedirs(result_dir, exist_ok=True)
            self.update_status.emit(f"Detected {len(indexes)} peaks, saving results...")
            self.update_progress.emit(50)

            # 6. 处理每个尖峰
            total_peaks = len(indexes)
            for ind, peak_idx in enumerate(indexes):
                progress = 50 + int(40 * (ind + 1) / total_peaks)
                self.update_progress.emit(progress)
                self.update_status.emit(f"Processing peak {ind + 1}/{total_peaks}")

                # 截取波形
                start_idx = max(0, peak_idx - weith)
                end_idx = min(len(df), peak_idx + weith)
                peak_df = df.iloc[start_idx:end_idx].copy()

                # 保存子文件
                sub_file_name = f"{os.path.splitext(base_file_name)[0]}_{ind}.csv"
                sub_file_path = os.path.join(result_dir, sub_file_name)
                peak_df.to_csv(sub_file_path, index=False)
                self.processed_files.append(sub_file_path)

                # 计算参数
                peak_df.loc[:, channel_a_col] = peak_df[channel_a_col] / 200.0
                peak_df.loc[:, channel_a_col] = savgol_filter(peak_df[channel_a_col], 20, 1, mode='interp')
                maxnum = peak_df[channel_a_col].iloc[peak_idx - start_idx]
                max_num_01 = maxnum * 0.1
                max_num_09 = maxnum * 0.9
                time_vals = peak_df[time_col].values
                channel_vals = peak_df[channel_a_col].values
                peak_time = time_vals[peak_idx - start_idx]

                # 计算tr/tf/tw
                left_09 = np.argmin([abs(v - max_num_09) if t < peak_time else np.inf 
                                    for v, t in zip(channel_vals, time_vals)])
                right_09 = np.argmin([abs(v - max_num_09) if t > peak_time else np.inf 
                                     for v, t in zip(channel_vals, time_vals)])
                left_01 = np.argmin([abs(v - max_num_01) if t < peak_time else np.inf 
                                    for v, t in zip(channel_vals, time_vals)])
                right_01 = np.argmin([abs(v - max_num_01) if t > peak_time else np.inf 
                                     for v, t in zip(channel_vals, time_vals)])

                tr = time_vals[left_09] - time_vals[left_01]
                tf = time_vals[right_01] - time_vals[right_09]
                tw = time_vals[right_01] - time_vals[left_01]

                # 记录日志
                self.log_data[sub_file_name] = (round(maxnum, 4), round(max_num_09, 4), 
                                               round(max_num_01, 4), round(tr, 6), 
                                               round(tf, 6), round(tw, 6))
                with open(log_file_path, "a", encoding="utf-8") as f:
                    f.write(f"{sub_file_name},{maxnum},{max_num_09},{max_num_01},{tr},{tf},{tw}\n")

        except Exception as e:
            self.update_status.emit(f"Processing error: {str(e)}")
            raise

# -------------------------- 波形查看器（核心修复：Tensor转Python数值） --------------------------
class ImageViewer(QMainWindow):
    def __init__(self, file_list, log_data, model, class_indict, device):
        super().__init__()
        self.file_list = file_list
        self.log_data = log_data
        self.model = model
        self.class_indict = class_indict
        self.device = device
        self.current_idx = 0
        self.data_transform = get_data_transform()
        self.init_ui()
        if self.file_list:
            self.load_image(0)

    def init_ui(self):
        self.setWindowTitle("Waveform Viewer (with Classification)")
        self.setGeometry(200, 100, 1200, 900)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 1. 标题区域
        self.title_layout = QHBoxLayout()
        self.file_name_label = QLabel("File: ---", font=QFont("Arial", 12, QFont.Bold))
        self.class_result_label = QLabel("Classification: ---", font=QFont("Arial", 12, QFont.Bold))
        self.class_result_label.setStyleSheet("color: #2E86AB;")
        self.title_layout.addWidget(self.file_name_label)
        self.title_layout.addStretch()
        self.title_layout.addWidget(self.class_result_label)
        main_layout.addLayout(self.title_layout)

        # 2. 波形画布
        self.canvas = MplCanvas(self, width=12, height=6, dpi=100)
        main_layout.addWidget(self.canvas)

        # 3. 波形参数显示
        self.param_layout = QGridLayout()
        self.param_labels = [
            QLabel("Max Value (maxnum):"), QLabel("0.0000"),
            QLabel("90% Max (max_num_09):"), QLabel("0.0000"),
            QLabel("10% Max (max_num_01):"), QLabel("0.0000"),
            QLabel("Rise Time (tr):"), QLabel("0.000000"),
            QLabel("Fall Time (tf):"), QLabel("0.000000"),
            QLabel("Pulse Width (tw):"), QLabel("0.000000")
        ]
        for i, label in enumerate(self.param_labels):
            label.setFont(QFont("Arial", 10))
            if i % 2 == 0:
                label.setStyleSheet("font-weight: bold;")
                self.param_layout.addWidget(label, i//2, 0, alignment=Qt.AlignRight | Qt.AlignVCenter)
            else:
                self.param_layout.addWidget(label, i//2, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        main_layout.addLayout(self.param_layout)

        # 4. 导航按钮
        self.btn_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous", font=QFont("Arial", 11))
        self.next_btn = QPushButton("Next", font=QFont("Arial", 11))
        self.close_btn = QPushButton("Close", font=QFont("Arial", 11))
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.close_btn.clicked.connect(self.close)
        self.btn_layout.addWidget(self.prev_btn)
        self.btn_layout.addWidget(self.next_btn)
        self.btn_layout.addStretch()
        self.btn_layout.addWidget(self.close_btn)
        main_layout.addLayout(self.btn_layout)

        self.update_buttons()

    def load_image(self, index):
        if not (0 <= index < len(self.file_list)):
            return

        current_file = self.file_list[index]
        current_file_name = os.path.basename(current_file)
        self.current_idx = index

        try:
            # -------------------------- 1. 绘制波形 --------------------------
            df = pd.read_csv(current_file, low_memory=False)
            df = df.iloc[1:,:].copy()
            df = pd.DataFrame(df, dtype='float')

            time_col = next((c for c in df.columns if 'time' in str(c).lower() or '时间' in str(c)), None)
            channel_a_col = next((c for c in df.columns if 'a' in str(c).lower() or '通道a' in str(c).lower()), None)
            if not time_col or not channel_a_col:
                raise ValueError("Missing time/channel A columns")

            # 绘制波形
            self.canvas.axes.clear()
            x = df[time_col]
            y = df[channel_a_col]
            self.canvas.axes.plot(x, y, marker='+', alpha=0.8, linewidth=1.2, color='#E63946')
            self.canvas.axes.set_title(f"Waveform: {os.path.splitext(current_file_name)[0]}", fontsize=14, pad=20)
            self.canvas.axes.set_xlabel("Time", fontsize=12, labelpad=10)
            self.canvas.axes.set_ylabel("Channel A", fontsize=12, labelpad=10)
            self.canvas.axes.grid(True, alpha=0.3, linestyle='--')
            self.canvas.axes.autoscale_view()
            self.canvas.draw()  # 刷新画布

            # -------------------------- 2. 内存缓冲区处理 --------------------------
            buffer = BytesIO()
            self.canvas.fig.savefig(
                buffer, 
                format='png', 
                dpi=self.canvas.fig.dpi,
                bbox_inches='tight'
            )
            buffer.seek(0)
            
            with Image.open(buffer) as img:
                img = img.convert('RGB')
                img_copy = img.copy()
            buffer.close()

            # -------------------------- 3. 模型分类（核心修复） --------------------------
            img_tensor = self.data_transform(img_copy).unsqueeze(0)
            with torch.no_grad():
                output = torch.squeeze(self.model(img_tensor.to(self.device))).cpu()
                predict_prob = torch.softmax(output, dim=0)
                predict_cls = torch.argmax(predict_prob).item()  # 转换为Python整数

            # 关键修复：用.item()将Tensor转换为Python float后再使用round()
            cls_prob = round(predict_prob[predict_cls].item() * 100, 1)

            # 显示分类结果
            cls_name = self.class_indict.get(str(predict_cls), "Unknown")
            self.class_result_label.setText(f"Classification: {cls_name} ({cls_prob}%)")
            if cls_name in ["脉冲", "Pulse"]:
                self.class_result_label.setStyleSheet("color: #2A9D8F; font-weight: bold;")
            else:
                self.class_result_label.setStyleSheet("color: #E76F51; font-weight: bold;")

            # -------------------------- 4. 显示波形参数 --------------------------
            if current_file_name in self.log_data:
                maxnum, max_num_09, max_num_01, tr, tf, tw = self.log_data[current_file_name]
                self.param_labels[1].setText(f"{maxnum}")
                self.param_labels[3].setText(f"{max_num_09}")
                self.param_labels[5].setText(f"{max_num_01}")
                self.param_labels[7].setText(f"{tr}")
                self.param_labels[9].setText(f"{tf}")
                self.param_labels[11].setText(f"{tw}")
            else:
                for i in [1,3,5,7,9,11]:
                    self.param_labels[i].setText("No data")

            # -------------------------- 5. 更新界面 --------------------------
            self.file_name_label.setText(f"File: {current_file_name}")
            self.update_buttons()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load waveform: {str(e)}")

    def prev_image(self):
        if self.current_idx > 0:
            self.load_image(self.current_idx - 1)

    def next_image(self):
        if self.current_idx < len(self.file_list) - 1:
            self.load_image(self.current_idx + 1)

    def update_buttons(self):
        self.prev_btn.setEnabled(self.current_idx > 0)
        self.next_btn.setEnabled(self.current_idx < len(self.file_list) - 1)

# -------------------------- 主窗口 --------------------------
class WaveformProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        # 初始化模型
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_indict = None
        self.init_model()
        # 存储处理结果
        self.processed_files = []
        self.log_data = {}
        self.image_viewer = None

    def init_ui(self):
        self.setWindowTitle("Waveform Processing & Classification Tool")
        self.setGeometry(100, 100, 900, 600)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 1. 阈值输入
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Waveform Threshold:", font=QFont("Arial", 11))
        self.threshold_input = QTextEdit("0", font=QFont("Arial", 11))
        self.threshold_input.setMaximumSize(120, 35)
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_input)
        threshold_layout.addStretch()
        main_layout.addLayout(threshold_layout)

        # 2. 文件选择
        self.select_btn = QPushButton("Select CSV File", font=QFont("Arial", 11))
        self.select_btn.clicked.connect(self.select_file)
        main_layout.addWidget(self.select_btn)

        # 3. 文件路径显示
        self.file_path_label = QLabel("No file selected", font=QFont("Arial", 10))
        self.file_path_label.setWordWrap(True)
        main_layout.addWidget(self.file_path_label)

        # 4. 功能按钮
        action_layout = QHBoxLayout()
        self.process_btn = QPushButton("Start Processing", font=QFont("Arial", 11))
        self.view_btn = QPushButton("View Waveforms", font=QFont("Arial", 11))
        self.process_btn.clicked.connect(self.start_processing)
        self.view_btn.clicked.connect(self.view_images)
        self.process_btn.setEnabled(False)
        self.view_btn.setEnabled(False)
        action_layout.addWidget(self.process_btn)
        action_layout.addWidget(self.view_btn)
        main_layout.addLayout(action_layout)

        # 5. 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # 6. 状态日志
        self.status_text = QTextEdit(font=QFont("Arial", 10))
        self.status_text.setReadOnly(True)
        self.status_text.setPlaceholderText("Processing logs will be displayed here...")
        main_layout.addWidget(self.status_text)

        self.selected_file = None

    def init_model(self):
        try:
            self.status_text.append(f"Initializing model on {self.device}...")
            self.model, self.class_indict = init_waveform_model(self.device)
            self.status_text.append("Model initialized successfully!")
        except Exception as e:
            self.status_text.append(f"Model initialization failed: {str(e)}")
            QMessageBox.warning(self, "Model Error", f"Cannot load model: {str(e)}")

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.selected_file = file_path
            self.file_path_label.setText(f"Selected File: {os.path.basename(file_path)}")
            self.process_btn.setEnabled(True)
            self.status_text.append(f"Selected file: {file_path}")

    def start_processing(self):
        try:
            threshold = float(self.threshold_input.toPlainText().strip())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter a valid number for threshold")
            return

        self.select_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.view_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_text.append("Starting processing...")

        self.process_thread = ProcessingThread(self.selected_file, threshold)
        self.process_thread.update_progress.connect(self.update_progress)
        self.process_thread.update_status.connect(self.update_status)
        self.process_thread.processing_finished.connect(self.processing_done)
        self.process_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, message):
        self.status_text.append(message)
        self.status_text.moveCursor(self.status_text.textCursor().End)

    def processing_done(self, processed_files, log_data):
        self.processed_files = processed_files
        self.log_data = log_data
        self.select_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        if processed_files:
            self.view_btn.setEnabled(True)
            QMessageBox.information(self, "Processing Done", 
                                   f"Generated {len(processed_files)} waveform files!\nCheck 'result' folder and log.csv")
        else:
            QMessageBox.information(self, "Processing Done", "No waveform files generated (no peaks detected)")

    def view_images(self):
        if not self.model or not self.class_indict:
            QMessageBox.warning(self, "Model Error", "Model is not initialized. Cannot classify waveforms.")
            return
        if not self.processed_files:
            QMessageBox.warning(self, "No Data", "No waveform files to view. Please process first.")
            return

        if self.image_viewer and self.image_viewer.isVisible():
            self.image_viewer.close()

        self.image_viewer = ImageViewer(
            file_list=self.processed_files,
            log_data=self.log_data,
            model=self.model,
            class_indict=self.class_indict,
            device=self.device
        )
        self.image_viewer.show()

# -------------------------- 程序入口 --------------------------
if __name__ == "__main__":
    matplotlib.use('Qt5Agg')
    plt.switch_backend('Qt5Agg')
    app = QApplication(sys.argv)
    window = WaveformProcessor()
    window.show()
    sys.exit(app.exec_())
    