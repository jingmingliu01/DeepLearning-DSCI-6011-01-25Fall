# YOLACT++ Campus Objects - Complete Usage Guide

**完整使用教程 - 从数据收集到模型部署**

---

## 📋 目录

1. [环境准备](#1-环境准备)
2. [数据收集与标注](#2-数据收集与标注)
3. [数据准备](#3-数据准备)
4. [模型训练](#4-模型训练)
5. [模型评估](#5-模型评估)
6. [推理测试](#6-推理测试)
7. [Web应用部署](#7-web应用部署)
8. [常见问题](#8-常见问题)

---

## 1. 环境准备

### 1.1 系统要求

**硬件要求：**
- CPU: 多核处理器（建议4核以上）
- RAM: 16GB+ （最低8GB）
- GPU: NVIDIA GPU with 8GB+ VRAM （推荐）
- 存储: 20GB+ 可用空间

**软件要求：**
- Python 3.7-3.9
- CUDA 11.0+ （如果使用GPU）
- Git

### 1.2 创建Python环境

使用conda（推荐）：

```bash
# 创建新环境
conda create -n yolact python=3.8

# 激活环境
conda activate yolact
```

或使用virtualenv：

```bash
# 创建虚拟环境
python -m venv venv

# 激活环境
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 1.3 安装PyTorch

访问 https://pytorch.org/get-started/locally/ 选择适合你系统的命令。

**示例（CUDA 11.8）：**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CPU版本（如果没有GPU）：**
```bash
pip install torch torchvision
```

### 1.4 安装项目依赖

```bash
cd Project/
pip install -r requirements.txt
```

### 1.5 克隆YOLACT++代码

```bash
cd Project/
git clone https://github.com/dbolya/yolact.git
```

### 1.6 下载预训练权重

**方法1：使用wget（推荐）**
```bash
cd weights/
wget -O yolact_plus_resnet50_54_800000.pth \
    "https://huggingface.co/dbolya/yolact-plus-resnet50/resolve/main/yolact_plus_resnet50_54_800000.pth?download=true"
cd ..
```

**方法2：使用curl**
```bash
cd weights/
curl -L -o yolact_plus_resnet50_54_800000.pth \
    "https://huggingface.co/dbolya/yolact-plus-resnet50/resolve/main/yolact_plus_resnet50_54_800000.pth?download=true"
cd ..
```

**方法3：手动下载**
1. 访问: https://huggingface.co/dbolya/yolact-plus-resnet50
2. 点击 Files and versions → yolact_plus_resnet50_54_800000.pth
3. 点击下载图标
4. 放到 `Project/weights/` 目录

**注意**：预训练权重已从Google Drive迁移到HuggingFace平台。旧的Google Drive链接已失效。

### 1.7 验证安装

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python config.py
```

应该看到配置信息输出，表示安装成功。

---

## 2. 数据收集与标注

### 2.1 拍摄照片

**目标：**收集200-300张包含以下物体的校园照片：
- 白板 (Whiteboard)
- 饮水机 (DrinkingWaterFountain)
- 大学标志 (UniversityLogo)

**拍摄要求：**
- 分辨率：1920x1080或更高
- 格式：JPG或PNG
- 多样性：
  - 不同时间（早晨、下午、晚上）
  - 不同地点（教室、走廊、室外）
  - 不同角度（正面、侧面、远近）
  - 不同光照条件

**文件命名建议：**
- `IMG_0001.jpg`, `IMG_0002.jpg`, ...
- 或 `whiteboard_001.jpg`, `fountain_001.jpg`, ...

**存放位置：**
将所有原始照片放到：`Project/data/raw_images/`

```bash
# 创建目录（如果不存在）
mkdir -p data/raw_images/

# 复制照片
cp /path/to/your/photos/* data/raw_images/
```

### 2.2 使用CVAT标注

#### 2.2.1 启动CVAT

**选项A：使用在线版本（推荐）**
访问: https://app.cvat.ai 并注册账号

**选项B：本地Docker版本**
```bash
docker run -d -p 8080:8080 openvino/cvat
# 访问 http://localhost:8080
```

#### 2.2.2 创建项目

1. 点击 **Projects** → **Create new project**
2. 填写：
   - Name: `Campus Objects`
   - Labels: 添加3个标签
     - `Whiteboard`
     - `DrinkingWaterFountain`
     - `UniversityLogo`

#### 2.2.3 创建任务并上传图片

1. 在项目中点击 **Tasks** → **Create new task**
2. 填写：
   - Name: `Campus Dataset`
   - Select files: 上传你的照片
3. 点击 **Submit**

#### 2.2.4 开始标注

1. 点击任务进入标注界面
2. 对每张图片：
   - 选择 **Polygon** 工具
   - 选择对应的标签（Whiteboard/DrinkingWaterFountain/UniversityLogo）
   - 沿着物体边缘点击创建多边形（15-25个点）
   - 按 **N** 键或点击第一个点完成多边形
   - 重复标注同一图片中的其他物体实例

**标注技巧：**
- 使用滚轮缩放，精确标注
- 每个物体实例都需要单独标注
- 确保多边形紧贴物体边缘
- 不要遗漏部分可见的物体

#### 2.2.5 导出标注

1. 回到任务列表
2. 点击任务右侧的 **⋮** → **Export task dataset**
3. 选择格式：**COCO 1.0**
4. 点击 **Export** 并下载

#### 2.2.6 整理标注文件

```bash
# 解压下载的ZIP文件
unzip annotations.zip

# 创建标注目录
mkdir -p data/coco_annotations/

# 复制标注文件
cp annotations/instances_default.json data/coco_annotations/instances.json
```

**检查标注文件结构：**
```bash
# 查看标注文件
head -50 data/coco_annotations/instances.json
```

应该包含 `images`, `annotations`, `categories` 三个主要字段。

---

## 3. 数据准备

### 3.1 运行数据准备脚本

```bash
cd Project/

# 运行准备脚本
python scripts/prepare_dataset.py
```

**脚本功能：**
- ✓ 验证COCO标注文件格式
- ✓ 统计数据集信息
- ✓ 自动划分训练/验证/测试集（70%/20%/10%）
- ✓ 调整图片大小到550x550
- ✓ 生成处理后的数据集

### 3.2 检查处理结果

```bash
# 查看数据集信息
cat data/dataset_info.json

# 检查生成的文件
ls data/processed/train/
ls data/processed/val/
ls data/processed/test/
```

**期望输出：**
```
data/processed/
├── train/
│   ├── annotations.json
│   ├── IMG_0001.jpg
│   ├── IMG_0002.jpg
│   └── ...
├── val/
│   ├── annotations.json
│   └── ...
└── test/
    ├── annotations.json
    └── ...
```

---

## 4. 模型训练

### 4.1 配置检查

编辑 `config.py` 如果需要调整参数：

```python
# 训练参数
BATCH_SIZE = 8           # 如果GPU内存不足，改为4
NUM_EPOCHS = 50          # 训练轮数
LEARNING_RATE = 1e-3     # 学习率

# 类别（确保与你的标注一致）
CLASSES = [
    'Whiteboard',
    'DrinkingWaterFountain',
    'UniversityLogo'
]
```

### 4.2 开始训练

```bash
# 运行训练脚本
python scripts/train.py
```

**训练过程：**
1. 自动检查环境和依赖
2. 验证数据集和预训练权重
3. 配置YOLACT++数据集
4. 开始训练

**训练监控：**
训练过程中会显示：
- Loss值（应该逐渐下降）
- 学习率
- 每个epoch的时间
- 验证指标

**训练输出：**
- 模型检查点：`outputs/checkpoints/`
- 训练日志：`outputs/logs/`
- TensorBoard日志（可选）

### 4.3 使用TensorBoard监控（可选）

```bash
# 在另一个终端窗口
tensorboard --logdir=outputs/logs/

# 访问 http://localhost:6006
```

### 4.4 训练时间估计

**预期训练时间（50 epochs）：**
- GPU (RTX 3060/4060): 2-4小时
- GPU (RTX 2060): 4-6小时
- CPU: 不推荐（30+ 小时）

### 4.5 提前停止

如果发现loss不再下降，可以按 `Ctrl+C` 停止训练。最佳模型已自动保存。

---

## 5. 模型评估

### 5.1 运行评估脚本

```bash
# 在测试集上评估模型
python scripts/eval_model.py
```

**评估指标：**
- mAP (mean Average Precision)
- mAP@50, mAP@75
- 每个类别的AP

### 5.2 查看评估结果

```bash
# 查看评估日志
cat outputs/logs/eval_results.log

# 查看COCO评估结果
cat outputs/results/metrics.json
```

### 5.3 评估特定模型

```bash
# 评估指定的检查点
python scripts/eval_model.py --model outputs/checkpoints/yolact_base_50_12000.pth
```

---

## 6. 推理测试

### 6.1 单张图片推理

```bash
# 对单张图片进行推理
python scripts/inference.py --image path/to/test_image.jpg
```

**示例：**
```bash
python scripts/inference.py --image data/processed/test/IMG_0100.jpg
```

**输出：**
- 结果图片：`outputs/results/images/result_IMG_0100.jpg`
- 自动显示结果（可以用 `--no-display` 禁用）

### 6.2 批量推理

```bash
# 对文件夹中的所有图片进行推理
python scripts/inference.py --folder path/to/images/ --output outputs/results/batch/
```

### 6.3 指定模型

```bash
# 使用特定的模型权重
python scripts/inference.py --image test.jpg --model outputs/checkpoints/best_model.pth
```

---

## 7. Web应用部署

### 7.1 启动Web应用

```bash
cd web_app/
python app.py
```

**输出：**
```
✓ Model loaded: ../weights/campus_objects_best.pth

🚀 Starting Flask app on http://0.0.0.0:5000
   Press Ctrl+C to stop
```

### 7.2 使用Web界面

1. 打开浏览器访问: http://localhost:5000
2. 点击或拖拽上传图片
3. 点击 **Detect Objects**
4. 查看检测结果

### 7.3 局域网访问

如果想让其他设备访问：

1. 找到你的IP地址：
   ```bash
   # Linux/Mac
   ifconfig | grep "inet "

   # Windows
   ipconfig
   ```

2. 其他设备访问: `http://YOUR_IP:5000`

### 7.4 配置Web应用

编辑 `config.py`：

```python
# Web应用配置
FLASK_HOST = '0.0.0.0'    # 允许外部访问
FLASK_PORT = 5000         # 端口号
FLASK_DEBUG = False       # 生产环境设为False

# 推理参数
INFERENCE_SCORE_THRESHOLD = 0.5  # 置信度阈值（0-1）
TOP_K = 15                       # 最多检测数量
```

---

## 8. 常见问题

### 8.1 安装问题

**Q: 安装PyTorch时失败**
```bash
# 尝试使用conda安装
conda install pytorch torchvision cudatoolkit=11.8 -c pytorch
```

**Q: pycocotools安装失败**
```bash
# 先安装Cython
pip install cython
# 然后安装pycocotools
pip install pycocotools
```

**Q: OpenCV安装失败**
```bash
# 尝试使用conda
conda install opencv -c conda-forge
```

### 8.2 数据问题

**Q: COCO标注文件格式错误**

检查JSON文件结构：
```python
import json
with open('data/coco_annotations/instances.json') as f:
    data = json.load(f)
    print("Images:", len(data['images']))
    print("Annotations:", len(data['annotations']))
    print("Categories:", data['categories'])
```

**Q: 图片找不到**

确保图片路径正确：
```bash
# 检查图片是否在正确位置
ls data/raw_images/
```

### 8.3 训练问题

**Q: CUDA out of memory**

减小batch size：
```python
# 在config.py中
BATCH_SIZE = 4  # 或更小
```

**Q: 训练速度很慢**

1. 确认GPU正在使用：
   ```bash
   nvidia-smi
   ```

2. 减少worker数量：
   ```python
   # 在config.py中
   NUM_WORKERS = 2
   ```

**Q: Loss不下降**

1. 检查学习率是否太低
2. 确认数据集质量
3. 尝试训练更多epochs

### 8.4 推理问题

**Q: 检测不到物体**

降低置信度阈值：
```python
# 在config.py中
INFERENCE_SCORE_THRESHOLD = 0.3  # 从0.5降到0.3
```

**Q: 检测结果不准确**

1. 需要更多训练数据
2. 增加训练epochs
3. 改进标注质量

### 8.5 Web应用问题

**Q: 无法访问Web界面**

检查防火墙设置：
```bash
# Linux
sudo ufw allow 5000

# 或更换端口
# 在config.py中修改 FLASK_PORT
```

**Q: 推理超时**

增加超时时间：
```python
# 在web_app/app.py中
result = subprocess.run(cmd, timeout=60)  # 从30改到60秒
```

---

## 9. 项目文件清单

完成所有步骤后，你应该有以下文件结构：

```
Project/
├── config.py                          ✓ 配置文件
├── requirements.txt                   ✓ 依赖列表
├── USAGE_GUIDE.md                     ✓ 本文档
├── PROJECT_STRUCTURE.md               ✓ 项目结构
│
├── data/                              ✓ 数据目录
│   ├── raw_images/                    📸 你的原始照片
│   ├── coco_annotations/
│   │   └── instances.json             🏷️ CVAT导出的标注
│   ├── processed/                     ✓ 处理后的数据
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── dataset_info.json              ✓ 数据集统计
│
├── scripts/                           ✓ 所有脚本
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── eval_model.py
│   └── inference.py
│
├── yolact/                            ✓ YOLACT++代码
├── weights/
│   └── yolact_plus_resnet50_54_800000.pth  💾 预训练权重
│
├── outputs/                           ✓ 训练输出
│   ├── checkpoints/                   💾 模型检查点
│   ├── logs/                          📊 训练日志
│   └── results/                       🖼️ 推理结果
│
└── web_app/                           ✓ Web应用
    ├── app.py
    ├── templates/
    │   └── index.html
    └── static/
```

---

## 10. 完整工作流程总结

### 阶段1：准备（1-2天）
```bash
# 1. 安装环境
conda create -n yolact python=3.8
conda activate yolact

# 2. 安装PyTorch（根据你的系统选择）
# GPU版本（CUDA 11.8）：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# 或 CPU版本：
# pip install torch torchvision

# 3. 安装其他依赖
pip install -r requirements.txt

# 4. 克隆YOLACT++
git clone https://github.com/dbolya/yolact.git

# 5. 下载预训练权重（从HuggingFace）
cd weights/
wget -O yolact_plus_resnet50_54_800000.pth \
    "https://huggingface.co/dbolya/yolact-plus-resnet50/resolve/main/yolact_plus_resnet50_54_800000.pth?download=true"
cd ..
```

### 阶段2：数据收集与标注（1-2周）
```bash
# 1. 拍摄200-300张照片
# 2. 上传到CVAT并标注
# 3. 导出COCO格式标注
# 4. 整理文件到data/目录
```

### 阶段3：训练（1天）
```bash
# 1. 准备数据
python scripts/prepare_dataset.py

# 2. 训练模型
python scripts/train.py

# 等待2-6小时（取决于GPU）
```

### 阶段4：评估和部署（1天）
```bash
# 1. 评估模型
python scripts/eval_model.py

# 2. 测试推理
python scripts/inference.py --image test.jpg

# 3. 启动Web应用
python web_app/app.py
```

---

## 11. 项目展示建议

### 11.1 准备演示材料

1. **演示视频**：
   - 录制Web应用使用过程
   - 展示不同场景下的检测效果

2. **结果图片**：
   - 准备10-20张最佳检测结果
   - 包含不同物体类别

3. **性能指标**：
   - 训练曲线（loss vs epochs）
   - mAP分数
   - 推理速度（FPS）

### 11.2 演示文稿大纲

1. **项目背景**
   - 实例分割任务介绍
   - 选择YOLACT++的原因

2. **数据集**
   - 数据收集过程
   - 标注示例
   - 数据集统计

3. **模型和方法**
   - YOLACT++架构
   - 迁移学习策略（层冻结）
   - 训练配置

4. **实验结果**
   - 定量结果（mAP等）
   - 定性结果（可视化）
   - 与baseline对比

5. **实时演示**
   - Web应用展示
   - 现场检测

6. **总结与展望**
   - 项目成果
   - 遇到的挑战
   - 未来改进方向

---

## 12. 技术支持

**遇到问题？**

1. 查看本文档的[常见问题](#8-常见问题)部分
2. 检查配置文件 `config.py`
3. 查看日志文件 `outputs/logs/`
4. 参考YOLACT++官方文档: https://github.com/dbolya/yolact

**联系方式：**
- 项目GitHub Issues
- 课程讨论区

---

## 13. 许可和引用

**YOLACT++论文引用：**
```bibtex
@article{bolya2020yolact++,
  title={YOLACT++: Better Real-time Instance Segmentation},
  author={Bolya, Daniel and Zhou, Chong and Xiao, Fanyi and Lee, Yong Jae},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020}
}
```

**代码许可：**
- YOLACT++: MIT License
- 本项目代码: 用于教育目的

---

**🎉 祝你项目顺利完成！Good Luck!**
