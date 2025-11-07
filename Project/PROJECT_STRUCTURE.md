# YOLACT++ Campus Objects - Project Structure

## Directory Structure

```
Project/
├── README.md                          # 项目概述
├── USAGE_GUIDE.md                     # 完整使用教程（重要！）
├── PROJECT_STRUCTURE.md               # 本文件：项目结构说明
├── requirements.txt                   # Python依赖
├── config.py                          # 配置文件
│
├── data/                              # 数据目录
│   ├── raw_images/                    # 原始照片（你拍摄的照片放这里）
│   ├── coco_annotations/              # CVAT导出的COCO标注
│   │   └── instances.json             # 完整的COCO标注文件
│   ├── processed/                     # 处理后的数据
│   │   ├── train/                     # 训练集
│   │   ├── val/                       # 验证集
│   │   └── test/                      # 测试集
│   └── dataset_info.json              # 数据集统计信息
│
├── scripts/                           # 脚本目录
│   ├── prepare_dataset.py             # 数据预处理
│   ├── dataset.py                     # 数据集类
│   ├── train.py                       # 训练脚本
│   ├── eval_model.py                  # 评估脚本
│   └── inference.py                   # 推理脚本
│
├── yolact/                            # YOLACT++代码（克隆自官方repo）
│   ├── data/
│   ├── layers/
│   ├── utils/
│   ├── train.py
│   └── eval.py
│
├── weights/                           # 模型权重目录
│   ├── yolact_plus_resnet50_54_800000.pth  # 预训练权重
│   └── campus_objects_best.pth        # 训练后的最佳模型
│
├── outputs/                           # 输出目录
│   ├── logs/                          # 训练日志
│   ├── checkpoints/                   # 训练检查点
│   └── results/                       # 推理结果
│       ├── images/                    # 可视化结果图片
│       └── metrics.json               # 评估指标
│
└── web_app/                           # Web应用
    ├── app.py                         # Flask应用
    ├── templates/                     # HTML模板
    │   └── index.html
    ├── static/                        # 静态文件
    │   ├── css/
    │   ├── js/
    │   └── uploads/                   # 上传的图片
    └── requirements.txt               # Web应用依赖
```

## File Descriptions

### Configuration Files
- **config.py**: 中心化配置文件，包含所有超参数、路径、类别定义等
- **requirements.txt**: 项目依赖包列表

### Data Files
- **data/raw_images/**: 存放你拍摄的原始照片
- **data/coco_annotations/instances.json**: 从CVAT导出的COCO格式标注
- **data/processed/**: 自动生成的训练/验证/测试集

### Script Files
- **prepare_dataset.py**: 数据预处理，包括划分数据集、验证标注、统计信息
- **dataset.py**: PyTorch Dataset类，用于加载数据
- **train.py**: 模型训练主脚本
- **eval_model.py**: 模型评估脚本
- **inference.py**: 单张图片推理脚本

### Model Files
- **yolact/**: YOLACT++官方代码（需要克隆）
- **weights/**: 预训练权重和训练后的模型

### Output Files
- **outputs/logs/**: TensorBoard日志和训练日志
- **outputs/checkpoints/**: 训练过程中的检查点
- **outputs/results/**: 评估结果和可视化图片

### Web Application
- **web_app/app.py**: Flask web应用，用于演示模型

## Quick Start

1. 按照 `USAGE_GUIDE.md` 安装环境
2. 将标注好的数据放入 `data/` 目录
3. 运行 `python scripts/prepare_dataset.py` 准备数据
4. 运行 `python scripts/train.py` 训练模型
5. 运行 `python scripts/inference.py` 测试推理
6. 运行 `python web_app/app.py` 启动Web应用

详细步骤请参考 **USAGE_GUIDE.md**
