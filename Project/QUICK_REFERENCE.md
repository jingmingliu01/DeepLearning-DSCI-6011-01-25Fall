# 🚀 Quick Reference Card

**YOLACT++ Campus Objects - 常用命令速查**

---

## 📦 环境设置

```bash
# 创建环境
conda create -n yolact python=3.8
conda activate yolact

# 或使用自动设置脚本
bash setup.sh

# 安装依赖
pip install -r requirements.txt

# 克隆YOLACT++
git clone https://github.com/dbolya/yolact.git

# 下载预训练权重
pip install gdown
gdown 1Uww4nwh1FJE9L9fGPVUcPMLS7_qXj7JX -O weights/yolact_plus_resnet50_54_800000.pth
```

---

## 📸 数据准备

```bash
# 1. 将照片放到
data/raw_images/

# 2. 将CVAT导出的标注放到
data/coco_annotations/instances.json

# 3. 运行数据准备
python scripts/prepare_dataset.py
```

---

## 🎯 训练

```bash
# 开始训练
python scripts/train.py

# 监控训练（可选）
tensorboard --logdir=outputs/logs/
```

---

## 📊 评估

```bash
# 评估模型
python scripts/eval_model.py

# 评估指定模型
python scripts/eval_model.py --model outputs/checkpoints/model.pth
```

---

## 🖼️ 推理

```bash
# 单张图片
python scripts/inference.py --image test.jpg

# 文件夹批量处理
python scripts/inference.py --folder images/ --output results/

# 不显示结果窗口
python scripts/inference.py --image test.jpg --no-display

# 使用特定模型
python scripts/inference.py --image test.jpg --model weights/best.pth
```

---

## 🌐 Web应用

```bash
# 启动Web服务
cd web_app/
python app.py

# 访问
http://localhost:5000

# 允许局域网访问
# 编辑 config.py:
# FLASK_HOST = '0.0.0.0'
```

---

## ⚙️ 配置文件

编辑 `config.py` 修改设置：

```python
# 类别
CLASSES = ['Whiteboard', 'DrinkingWaterFountain', 'UniversityLogo']

# 训练参数
BATCH_SIZE = 8          # GPU内存不足改为4
NUM_EPOCHS = 50         # 训练轮数
LEARNING_RATE = 1e-3    # 学习率

# 数据集划分
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# 推理阈值
INFERENCE_SCORE_THRESHOLD = 0.5  # 降低可检测更多物体
```

---

## 🔧 常见问题快速修复

### GPU内存不足
```python
# config.py
BATCH_SIZE = 4  # 或更小
```

### 检测不到物体
```python
# config.py
INFERENCE_SCORE_THRESHOLD = 0.3  # 降低阈值
```

### 训练太慢
```bash
# 检查GPU是否在使用
nvidia-smi

# 减少worker
# config.py
NUM_WORKERS = 2
```

### pycocotools安装失败
```bash
pip install cython
pip install pycocotools
```

---

## 📁 重要文件路径

```
数据：
  data/raw_images/                    # 原始照片
  data/coco_annotations/instances.json # COCO标注
  data/processed/                     # 处理后数据

模型：
  weights/yolact_plus_*.pth          # 预训练权重
  outputs/checkpoints/*.pth          # 训练检查点

结果：
  outputs/results/images/            # 推理结果图片
  outputs/logs/                      # 训练日志
```

---

## 📚 文档

- **USAGE_GUIDE.md** - 完整教程 ⭐
- **README.md** - 项目概述
- **PROJECT_STRUCTURE.md** - 项目结构
- **CVAT_Annotation_Tutorial.md** - 标注教程

---

## 🐛 调试技巧

```bash
# 检查Python环境
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# 检查配置
python config.py

# 查看数据集信息
cat data/dataset_info.json | python -m json.tool

# 查看训练日志
tail -f outputs/logs/train.log

# 测试YOLACT导入
cd yolact && python -c "import data.config"
```

---

## ⏱️ 时间估算

| 任务 | 时间 |
|------|------|
| 环境安装 | 30分钟 |
| 数据收集 | 2-4小时 |
| 数据标注 | 40-60小时 (200-300张) |
| 数据准备 | 5-10分钟 |
| 模型训练 | 2-6小时 (GPU) |
| 模型评估 | 10-20分钟 |
| Web部署 | 5分钟 |

---

## 🎯 完整工作流程

```bash
# 1. 设置环境
bash setup.sh

# 2. 收集数据（手动）
# 拍摄200-300张照片
# 使用CVAT标注

# 3. 准备数据
python scripts/prepare_dataset.py

# 4. 训练模型
python scripts/train.py

# 5. 评估模型
python scripts/eval_model.py

# 6. 测试推理
python scripts/inference.py --image test.jpg

# 7. 启动Web应用
python web_app/app.py
```

---

## 💡 提示

- 训练时定期保存检查点（自动）
- 使用TensorBoard监控训练
- 降低阈值可以检测更多物体（但可能增加误检）
- 提高阈值可以提高精度（但可能漏检）
- 数据质量 > 数据数量
- 标注要准确，不要着急

---

**📞 需要帮助？查看 USAGE_GUIDE.md 的详细说明！**
