# Wave Tool 波形处理与分类工具 README文档

## 一、项目简介
Wave Tool 是一款基于 Python + PyQt5 + 深度学习的波形数据处理工具，支持从 CSV 波形文件中提取尖峰波形、计算波形参数（峰值、上升时间、脉宽等），并通过 Swin-Tiny 模型自动分类波形是否为脉冲信号，适用于波形数据的快速分析与验证。

### 核心功能
1. **波形数据处理**：读取 CSV 格式波形数据，检测尖峰并截取波形片段
2. **参数计算**：自动计算波形的峰值（maxnum）、90%峰值（max_num_09）、10%峰值（max_num_01）、上升时间（tr）、下降时间（tf）、脉宽（tw）
3. **AI 分类**：基于 Swin-Tiny 模型自动判断波形是否为脉冲信号，显示分类概率
4. **可视化查看**：支持波形图像浏览、分类结果与参数实时显示
5. **可执行打包**：支持打包为 Ubuntu 可执行文件，脱离 Python 环境运行


## 二、环境准备（基于 Conda）
### 1. 安装 Conda（若未安装）
- 下载地址：[Anaconda 官网](https://www.anaconda.com/products/distribution#download-section)（选择 Linux 版本）
- 安装命令：`bash Anaconda3-xxxx-Linux-x86_64.sh`（按提示完成安装，重启终端生效）


### 2. 创建并激活 Conda 环境
打开终端，执行以下命令创建专属环境（Python 版本建议 3.10，避免兼容性问题）：
```bash
# 创建环境（名称：wave_env，Python 3.10）
conda create -n wave_env python=3.10 -y

# 激活环境（每次运行程序前需执行）
conda activate wave_env
```


### 3. 安装依赖库
#### 3.1 Ubuntu 系统依赖（必装，解决 GUI 与打包问题）
```bash
# 安装 PyQt5 与 XML 解析依赖（解决 pyexpat 错误）
sudo apt update
sudo apt install -y libexpat1 libexpat1-dev python3-pyqt5 python3-pyqt5.qtmultimedia libxcb-xinerama0 libxcb-render0
```

#### 3.2 Python 依赖（通过 pip/conda 安装）
```bash
# 基础数据处理库
pip install pandas numpy scipy peakutils chardet

# GUI 与可视化库
pip install matplotlib PyQt5 pillow

# 深度学习库（Torch + TorchVision，根据 GPU 情况选择）
# 情况1：有 NVIDIA GPU（需提前安装 CUDA，示例为 CUDA 11.8）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 情况2：无 GPU（安装 CPU 版本）
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 打包工具
pip install pyinstaller
```


## 三、文件准备
将以下文件放在同一工作目录（例如 `~/Desktop/wave`），确保文件名与路径对应：
| 文件名                  | 说明                                  | 备注                                  |
|-------------------------|---------------------------------------|---------------------------------------|
| `waveform_processor.py` | 主程序文件                            | 核心代码，已处理路径与依赖问题        |
| `model.py`              | 模型定义文件                          | 包含 Swin-Tiny 模型（`swin_tiny_patch4_window7_224` 函数） |
| `model-2.pth`           | 模型权重文件                          | 预训练的波形分类模型                  |
| `class_indices.json`    | 类别映射文件                          | 定义分类标签（如 `{"0":"非脉冲","1":"脉冲"}`） |
| `wave.png`（可选）      | 程序图标                              | 用于桌面快捷方式显示                  |


## 四、程序运行
### 方式1：直接通过 Python 运行（适合调试）
1. 终端进入工作目录：
   ```bash
   cd ~/Desktop/wave
   ```
2. 激活 Conda 环境（若未激活）：
   ```bash
   conda activate wave_env
   ```
3. 运行程序：
   ```bash
   python waveform_processor.py
   ```


### 方式2：打包为可执行文件（适合最终使用）
#### 1. 清理旧打包产物（若之前打包过）
```bash
rm -rf build/ dist/ "Wave Tool.spec"
```

#### 2. 创建自定义打包配置文件（`wave.spec`）
在工作目录创建 `wave.spec` 文件，内容如下（直接复制）：
```python
# -*- mode: python ; coding: utf-8 -*-
block_cipher = None

a = Analysis(
    ['waveform_processor.py'],  # 主程序文件
    pathex=[],
    binaries=[],
    # 需打包的外部文件（路径：目标路径）
    datas=[
        ('model-2.pth', '.'),
        ('class_indices.json', '.'),
        ('model.py', '.')
    ],
    # 强制包含缺失的依赖模块
    hiddenimports=['xml.parsers.expat', 'pkg_resources.py2_warn'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Wave Tool',  # 可执行文件名
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # 隐藏终端黑框
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
```

#### 3. 执行打包命令
```bash
pyinstaller wave.spec
```

#### 4. 打包后运行
- 打包产物在 `dist` 文件夹中，可执行文件名为 `Wave Tool`
- 终端运行步骤：
  ```bash
  # 进入 dist 目录
  cd ~/Desktop/wave/dist
  # 赋予执行权限（首次运行必需）
  chmod +x "Wave Tool"
  # 运行程序
  ./"Wave Tool"
  ```

#### 5. 创建桌面快捷方式（带图标，推荐）
1. 在桌面右键 → 「新建文档」→ 「空文件」，命名为 `Wave Tool.desktop`
2. 打开文件，粘贴以下内容（**需修改 3 处路径为你的实际路径**）：
   ```ini
   [Desktop Entry]
   Name=Wave Tool
   Comment=波形处理与分类工具
   # 替换为你的可执行文件路径（如 /home/你的用户名/Desktop/wave/dist/Wave Tool）
   Exec=/home/cyrusliu/Desktop/wave/dist/Wave Tool
   # 替换为你的图标路径（如 /home/你的用户名/Desktop/wave/wave.png）
   Icon=/home/cyrusliu/Desktop/wave/wave.png
   Terminal=false  # 不显示终端
   Type=Application
   Categories=Utility;Application;  # 归类到“实用工具”
   ```
3. 保存文件后，右键该快捷方式 → 「属性」→ 「权限」→ 勾选「允许作为程序执行文件」
4. 双击桌面的 `Wave Tool` 图标即可启动程序


## 五、使用教程
### 1. 程序界面说明
启动后界面包含以下核心模块：
- **阈值输入框**：设置尖峰检测阈值（默认 0，可根据波形振幅调整，值越大检测越严格）
- **Select CSV File 按钮**：选择待处理的 CSV 波形文件
- **Start Processing 按钮**：开始处理数据（需先选择文件）
- **View Waveforms 按钮**：查看处理后的波形与分类结果（需先完成处理）
- **进度条**：显示数据处理进度
- **日志区**：显示操作日志（如文件选择、处理状态、错误信息）


### 2. 完整操作流程
#### 步骤1：选择 CSV 波形文件
1. 点击「Select CSV File」按钮
2. 在文件对话框中选择你的波形 CSV 文件（需包含「时间」列和「通道 A」列，中英文列名均支持）
3. 日志区会显示“Selected file: 路径”，表示文件选择成功

#### 步骤2：设置阈值并处理数据
1. 在阈值输入框中输入数值（例如 0.5，根据波形数据调整，建议从 0 开始测试）
2. 点击「Start Processing」按钮
3. 进度条开始推进，日志区显示处理状态（如“Detecting peaks...”“Processing peak 1/5”）
4. 处理完成后，会弹出提示框，显示生成的波形文件数量
   - 处理后的波形片段保存在 `result` 文件夹（与原 CSV 文件同目录）
   - 波形参数（maxnum、tr、tf 等）保存在 `log.csv` 文件（与原 CSV 文件同目录）

#### 步骤3：查看波形与分类结果
1. 点击「View Waveforms」按钮，打开波形查看窗口
2. 窗口包含以下信息：
   - **文件名**：当前查看的波形片段文件名
   - **分类结果**：显示模型预测的类别（如“脉冲 (98.5%)”），绿色为脉冲，红色为非脉冲
   - **波形图**：显示通道 A 的波形曲线
   - **波形参数**：显示 maxnum、max_num_09、max_num_01、tr、tf、tw 的具体数值
   - **导航按钮**：「Previous」（上一张）、「Next」（下一张）、「Close」（关闭窗口）
3. 操作：点击「Next」/「Previous」切换不同的波形片段，查看分类结果与参数


## 六、常见问题排查
### 1. 运行报错：`ImportError: undefined symbol: XML_SetReparseDeferralEnabled`
- 原因：缺少 `libexpat` 系统库
- 解决：执行以下命令安装，然后重新打包运行：
  ```bash
  sudo apt install -y libexpat1 libexpat1-dev
  ```

### 2. 模型加载失败：`Model initialization failed: 模型权重不存在`
- 原因：`model-2.pth` 文件路径错误或缺失
- 解决：
  1. 确认 `model-2.pth` 在工作目录中
  2. 检查代码中 `MODEL_WEIGHT_PATH = get_resource_path("model-2.pth")` 的文件名是否与实际一致

### 3. 波形加载失败：`Failed to load waveform: Missing time/channel A columns`
- 原因：CSV 文件中没有找到「时间」列或「通道 A」列
- 解决：
  1. 检查 CSV 文件的表头，确保包含“时间”“通道 A”（或英文“time”“channel A”）
  2. 若列名不同，可修改代码中 `time_col` 和 `channel_a_col` 的检测逻辑（见 `ProcessingThread` 类的 `process_file` 方法）

### 4. 打包后闪退，无任何提示
- 原因：依赖缺失或路径错误
- 解决：
  1. 去掉 `wave.spec` 中 `console=False` 改为 `console=True`，重新打包
  2. 终端运行 `./"Wave Tool"`，查看报错日志，定位缺失的依赖
  3. 根据日志提示，在 `hiddenimports` 中添加缺失的模块（如 `hiddenimports=['xxx', 'yyy']`）

### 5. Ubuntu 下图标不显示
- 原因：PyInstaller 在 Ubuntu 下不支持直接给可执行文件加图标
- 解决：通过「桌面快捷方式」显示图标（参考第四章第 5 节），确保 `Icon=` 后的路径正确


## 七、注意事项
1. **CSV 文件格式要求**：
   - 必须包含时间列（如“时间”“time”）和通道 A 列（如“通道 A”“channel A”）
   - 数据编码建议为 UTF-8 或 GBK，程序会自动检测编码
2. **模型兼容性**：
   - `model.py` 中的 `swin_tiny_patch4_window7_224` 函数需与 `model-2.pth` 的训练结构一致，否则模型加载失败
   - 若更换模型，需同步修改 `num_classes`（当前为 2 类：脉冲/非脉冲）
3. **阈值调整建议**：
   - 若检测不到尖峰，可降低阈值（如 0 → 0.1）
   - 若检测到过多杂波尖峰，可提高阈值（如 0.1 → 0.5）
4. **打包产物说明**：
   - `dist` 文件夹中的 `Wave Tool` 是独立可执行文件，可复制到其他 Ubuntu 系统运行（需安装相同的系统依赖）
   - `result` 文件夹和 `log.csv` 会生成在原 CSV 文件的同目录下，便于后续分析



若遇到其他问题，可通过以下方式排查：
1. 查看程序日志区的错误信息（主窗口下方的文本框）
2. 终端运行可执行文件，查看详细报错（将 `console=False` 改为 `console=True` 重新打包）
3. 确认所有依赖库版本与本文档一致（尤其是 Python 3.10、PyTorch 2.0+、PyInstaller 6.15+）
