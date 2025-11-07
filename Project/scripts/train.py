#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLACT++ Training Script
使用层冻结策略进行迁移学习
"""

import sys
import os
import argparse
from pathlib import Path
import subprocess

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

def check_yolact_installation():
    """检查YOLACT++是否已安装"""
    if not config.YOLACT_ROOT.exists():
        print(f"❌ Error: YOLACT++ not found at: {config.YOLACT_ROOT}")
        print("\n请先运行以下命令安装YOLACT++:")
        print(f"  cd {PROJECT_ROOT}")
        print("  git clone https://github.com/dbolya/yolact.git")
        return False

    return True

def check_pretrained_weights():
    """检查预训练权重是否存在"""
    if not config.PRETRAINED_WEIGHTS.exists():
        print(f"❌ Error: Pretrained weights not found at: {config.PRETRAINED_WEIGHTS}")
        print("\n请下载预训练权重:")
        print("  方法1: 直接下载")
        print("    https://drive.google.com/file/d/1Uww4nwh1FJE9L9fGPVUcPMLS7_qXj7JX/view")
        print(f"    然后放到: {config.WEIGHTS_DIR}/")
        print("\n  方法2: 使用gdown")
        print("    pip install gdown")
        print(f"    gdown 1Uww4nwh1FJE9L9fGPVUcPMLS7_qXj7JX -O {config.PRETRAINED_WEIGHTS}")
        return False

    return True

def check_dataset():
    """检查数据集是否已准备"""
    train_dir = config.PROCESSED_DATA_DIR / "train"
    val_dir = config.PROCESSED_DATA_DIR / "val"

    if not train_dir.exists() or not val_dir.exists():
        print(f"❌ Error: Processed dataset not found")
        print("\n请先运行数据准备脚本:")
        print("  python scripts/prepare_dataset.py")
        return False

    train_ann = train_dir / "annotations.json"
    val_ann = val_dir / "annotations.json"

    if not train_ann.exists() or not val_ann.exists():
        print(f"❌ Error: Annotation files not found")
        print("\n请先运行数据准备脚本:")
        print("  python scripts/prepare_dataset.py")
        return False

    return True

def create_yolact_dataset_config():
    """在YOLACT的config.py中创建数据集配置"""
    print("\n📝 Creating YOLACT++ dataset configuration...")

    yolact_config_file = config.YOLACT_ROOT / "data" / "config.py"

    if not yolact_config_file.exists():
        print(f"❌ Error: YOLACT config file not found: {yolact_config_file}")
        return False

    # 读取现有配置
    with open(yolact_config_file, 'r', encoding='utf-8') as f:
        existing_config = f.read()

    # 检查是否已经添加了campus配置
    if 'campus_objects_dataset' in existing_config:
        print("✓ Campus objects dataset config already exists")
        return True

    # 创建配置内容
    campus_config = f'''

# ==================== Campus Objects Dataset ====================
# Added by scripts/train.py

campus_objects_dataset = dataset_base.copy({{
    'name': 'Campus Objects',

    'train_images': '{config.PROCESSED_DATA_DIR / "train"}/',
    'train_info': '{config.PROCESSED_DATA_DIR / "train" / "annotations.json"}',

    'valid_images': '{config.PROCESSED_DATA_DIR / "val"}/',
    'valid_info': '{config.PROCESSED_DATA_DIR / "val" / "annotations.json"}',

    'has_gt': True,
    'class_names': {config.CLASSES},
}})

campus_objects_config = yolact_base_config.copy({{
    'name': 'campus_objects',
    'dataset': campus_objects_dataset,
    'num_classes': {config.NUM_CLASSES},
    'max_size': {config.IMAGE_SIZE},
}})

# ==================== End Campus Objects Config ====================
'''

    # 追加配置
    with open(yolact_config_file, 'a', encoding='utf-8') as f:
        f.write(campus_config)

    print("✓ Campus objects dataset config added to YOLACT++")
    return True

def build_train_command():
    """构建训练命令"""
    cmd = [
        'python', str(config.YOLACT_ROOT / 'train.py'),
        f'--config=campus_objects_config',
        f'--batch_size={config.BATCH_SIZE}',
        f'--lr={config.LEARNING_RATE}',
        f'--save_folder={config.CHECKPOINTS_DIR}/',
        f'--log_folder={config.LOGS_DIR}/',
        f'--resume={config.PRETRAINED_WEIGHTS}',
        '--save_interval=5000',
        '--validation_epoch=2',
    ]

    if config.USE_GPU:
        cmd.append(f'--cuda=True')

    return cmd

def run_training():
    """运行训练"""
    print("\n🚀 Starting training...")
    print(f"\n训练配置:")
    print(f"  - Batch Size: {config.BATCH_SIZE}")
    print(f"  - Learning Rate: {config.LEARNING_RATE}")
    print(f"  - Epochs: {config.NUM_EPOCHS}")
    print(f"  - Image Size: {config.IMAGE_SIZE}")
    print(f"  - Classes: {config.CLASSES}")
    print(f"  - GPU: {'Enabled' if config.USE_GPU else 'Disabled'}")

    cmd = build_train_command()

    print(f"\n执行命令:")
    print(f"  {' '.join(cmd)}")
    print("\n" + "="*60)

    try:
        # 切换到YOLACT目录
        os.chdir(config.YOLACT_ROOT)

        # 运行训练
        process = subprocess.run(cmd, check=True)

        print("\n" + "="*60)
        print("✅ Training completed successfully!")
        print(f"\n模型保存在: {config.CHECKPOINTS_DIR}")
        print(f"日志保存在: {config.LOGS_DIR}")

        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return False
    finally:
        # 返回原目录
        os.chdir(PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description='Train YOLACT++ on Campus Objects')
    parser.add_argument('--skip-checks', action='store_true',
                        help='Skip pre-training checks')
    args = parser.parse_args()

    print("\n" + "="*60)
    print(" YOLACT++ Training - Campus Objects")
    print("="*60)

    if not args.skip_checks:
        # 1. 检查YOLACT++
        print("\n1️⃣  Checking YOLACT++ installation...")
        if not check_yolact_installation():
            sys.exit(1)
        print("✓ YOLACT++ found")

        # 2. 检查预训练权重
        print("\n2️⃣  Checking pretrained weights...")
        if not check_pretrained_weights():
            sys.exit(1)
        print("✓ Pretrained weights found")

        # 3. 检查数据集
        print("\n3️⃣  Checking dataset...")
        if not check_dataset():
            sys.exit(1)
        print("✓ Dataset ready")

        # 4. 创建数据集配置
        print("\n4️⃣  Setting up dataset configuration...")
        if not create_yolact_dataset_config():
            sys.exit(1)

    # 5. 开始训练
    print("\n5️⃣  Starting training...")
    success = run_training()

    if success:
        print("\n" + "="*60)
        print(" Next Steps")
        print("="*60)
        print("\n1. 评估模型:")
        print("   python scripts/eval_model.py")
        print("\n2. 测试推理:")
        print("   python scripts/inference.py --image path/to/image.jpg")
        print("\n3. 启动Web应用:")
        print("   python web_app/app.py")
        print("="*60 + "\n")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
