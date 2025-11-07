#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLACT++ Campus Objects - Configuration File
集中管理所有配置参数
"""

import os
from pathlib import Path

# ==================== 项目路径配置 ====================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# 数据目录
DATA_ROOT = PROJECT_ROOT / "data"
RAW_IMAGES_DIR = DATA_ROOT / "raw_images"
COCO_ANNOTATIONS_DIR = DATA_ROOT / "coco_annotations"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"

# YOLACT代码目录
YOLACT_ROOT = PROJECT_ROOT / "yolact"

# 权重目录
WEIGHTS_DIR = PROJECT_ROOT / "weights"
PRETRAINED_WEIGHTS = WEIGHTS_DIR / "yolact_plus_resnet50_54_800000.pth"
BEST_WEIGHTS = WEIGHTS_DIR / "campus_objects_best.pth"

# 输出目录
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
RESULTS_DIR = OUTPUTS_DIR / "results"

# Web应用目录
WEB_APP_DIR = PROJECT_ROOT / "web_app"

# ==================== 数据集配置 ====================

# 类别定义（按照你的项目）
CLASSES = [
    'Whiteboard',              # 白板
    'DrinkingWaterFountain',   # 饮水机
    'UniversityLogo'           # 大学标志
]

NUM_CLASSES = len(CLASSES)

# COCO标注文件路径
COCO_ANNOTATION_FILE = COCO_ANNOTATIONS_DIR / "instances.json"

# 数据集划分比例
TRAIN_RATIO = 0.7    # 70% 训练集
VAL_RATIO = 0.2      # 20% 验证集
TEST_RATIO = 0.1     # 10% 测试集

# 随机种子（保证可复现性）
RANDOM_SEED = 42

# 图像尺寸（YOLACT++标准输入）
IMAGE_SIZE = 550

# 数据增强参数
DATA_AUGMENTATION = {
    'horizontal_flip': True,
    'brightness': 0.2,
    'contrast': 0.2,
    'saturation': 0.2,
    'hue': 0.05
}

# ==================== 模型配置 ====================

# 骨干网络
BACKBONE = 'resnet50'  # 可选: resnet50, resnet101

# 是否使用YOLACT++特性
USE_YOLACT_PLUS = True

# 层冻结配置（迁移学习关键）
FREEZE_LAYERS = {
    'backbone': True,        # 冻结ResNet骨干网络
    'fpn': True,            # 冻结特征金字塔网络
    'proto_net': True,      # 冻结原型生成网络
    'prediction_layers': {  # 预测层部分冻结
        'bbox': True,       # 冻结边界框预测
        'mask': True,       # 冻结掩码系数预测
        'class': False      # 不冻结分类层（需要训练）
    }
}

# ==================== 训练配置 ====================

# 训练超参数
BATCH_SIZE = 8           # 根据GPU内存调整（8GB显存建议4-8）
NUM_EPOCHS = 50          # 迁移学习通常30-50个epoch足够
LEARNING_RATE = 1e-3     # 由于只训练分类层，可以用较高学习率
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

# 学习率调度
LR_SCHEDULER = {
    'type': 'step',       # 可选: step, cosine, plateau
    'step_size': 10,      # 每10个epoch衰减
    'gamma': 0.1          # 衰减因子
}

# 优化器
OPTIMIZER = 'SGD'  # 可选: SGD, Adam

# GPU设置
USE_GPU = True
GPU_ID = 0  # 如果有多个GPU，指定使用哪个
NUM_WORKERS = 4  # 数据加载线程数

# 训练日志
LOG_INTERVAL = 10        # 每10个batch打印一次
SAVE_INTERVAL = 5        # 每5个epoch保存一次
EVAL_INTERVAL = 2        # 每2个epoch验证一次

# 早停策略
EARLY_STOPPING = {
    'enabled': True,
    'patience': 10,       # 连续10个epoch没提升则停止
    'min_delta': 0.001    # 最小提升阈值
}

# ==================== 评估配置 ====================

# 评估指标
EVAL_METRICS = ['bbox', 'segm']  # 边界框和分割掩码

# mAP计算
MAP_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# NMS（非极大值抑制）阈值
NMS_THRESHOLD = 0.5

# 置信度阈值
SCORE_THRESHOLD = 0.3

# 最大检测数量
MAX_DETECTIONS = 100

# ==================== 推理配置 ====================

# 推理时的置信度阈值
INFERENCE_SCORE_THRESHOLD = 0.5

# 推理时的NMS阈值
INFERENCE_NMS_THRESHOLD = 0.5

# Top-K检测
TOP_K = 15

# 可视化配置
VISUALIZATION = {
    'show_bbox': True,        # 显示边界框
    'show_mask': True,        # 显示分割掩码
    'show_score': True,       # 显示置信度
    'mask_alpha': 0.45,       # 掩码透明度
    'bbox_thickness': 2,      # 边界框线宽
    'font_scale': 0.5         # 字体大小
}

# 颜色映射（RGB）
CLASS_COLORS = {
    'Whiteboard': (255, 255, 255),           # 白色
    'DrinkingWaterFountain': (0, 191, 255),  # 天蓝色
    'UniversityLogo': (255, 215, 0)          # 金色
}

# ==================== Web应用配置 ====================

# Flask配置
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = False

# 上传文件配置
UPLOAD_FOLDER = WEB_APP_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# ==================== 辅助函数 ====================

def create_dirs():
    """创建必要的目录"""
    dirs = [
        DATA_ROOT,
        RAW_IMAGES_DIR,
        COCO_ANNOTATIONS_DIR,
        PROCESSED_DATA_DIR,
        PROCESSED_DATA_DIR / "train",
        PROCESSED_DATA_DIR / "val",
        PROCESSED_DATA_DIR / "test",
        WEIGHTS_DIR,
        OUTPUTS_DIR,
        LOGS_DIR,
        CHECKPOINTS_DIR,
        RESULTS_DIR,
        RESULTS_DIR / "images",
        WEB_APP_DIR / "static" / "uploads"
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

    print("✓ All directories created successfully!")

def print_config():
    """打印配置信息"""
    print("\n" + "="*60)
    print(" YOLACT++ Campus Objects - Configuration")
    print("="*60)
    print(f"\n📁 Project Root: {PROJECT_ROOT}")
    print(f"\n🎯 Classes ({NUM_CLASSES}):")
    for i, cls in enumerate(CLASSES, 1):
        print(f"   {i}. {cls}")
    print(f"\n🖼️  Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"📊 Dataset Split: Train {TRAIN_RATIO*100:.0f}% | Val {VAL_RATIO*100:.0f}% | Test {TEST_RATIO*100:.0f}%")
    print(f"\n🔧 Training Config:")
    print(f"   - Batch Size: {BATCH_SIZE}")
    print(f"   - Epochs: {NUM_EPOCHS}")
    print(f"   - Learning Rate: {LEARNING_RATE}")
    print(f"   - Backbone: {BACKBONE}")
    print(f"\n🧊 Frozen Layers:")
    print(f"   - Backbone: {'✓' if FREEZE_LAYERS['backbone'] else '✗'}")
    print(f"   - FPN: {'✓' if FREEZE_LAYERS['fpn'] else '✗'}")
    print(f"   - ProtoNet: {'✓' if FREEZE_LAYERS['proto_net'] else '✗'}")
    print(f"   - Classification Layer: {'✗ (Trainable)' if not FREEZE_LAYERS['prediction_layers']['class'] else '✓'}")
    print("="*60 + "\n")

# ==================== 验证配置 ====================

def validate_config():
    """验证配置是否有效"""
    errors = []

    # 检查比例和是否为1
    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-6:
        errors.append(f"Dataset split ratios must sum to 1.0, got {TRAIN_RATIO + VAL_RATIO + TEST_RATIO}")

    # 检查类别数量
    if NUM_CLASSES == 0:
        errors.append("NUM_CLASSES must be greater than 0")

    # 检查batch size
    if BATCH_SIZE <= 0:
        errors.append("BATCH_SIZE must be greater than 0")

    # 检查epoch数量
    if NUM_EPOCHS <= 0:
        errors.append("NUM_EPOCHS must be greater than 0")

    # 检查学习率
    if LEARNING_RATE <= 0:
        errors.append("LEARNING_RATE must be greater than 0")

    if errors:
        print("\n❌ Configuration Errors:")
        for error in errors:
            print(f"   - {error}")
        return False

    print("✓ Configuration validated successfully!")
    return True

# ==================== 环境变量覆盖 ====================

def load_env_overrides():
    """从环境变量加载配置覆盖"""
    global BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, GPU_ID

    if 'BATCH_SIZE' in os.environ:
        BATCH_SIZE = int(os.environ['BATCH_SIZE'])

    if 'NUM_EPOCHS' in os.environ:
        NUM_EPOCHS = int(os.environ['NUM_EPOCHS'])

    if 'LEARNING_RATE' in os.environ:
        LEARNING_RATE = float(os.environ['LEARNING_RATE'])

    if 'GPU_ID' in os.environ:
        GPU_ID = int(os.environ['GPU_ID'])

# 初始化时自动加载环境变量
load_env_overrides()

if __name__ == "__main__":
    # 测试配置
    print_config()
    validate_config()
    create_dirs()
