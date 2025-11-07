#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集准备脚本
功能：
1. 验证COCO标注文件
2. 划分训练/验证/测试集
3. 调整图片大小到目标分辨率
4. 生成数据集统计信息
"""

import sys
import json
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict
import random

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

def load_coco_annotations(annotation_file):
    """加载COCO标注文件"""
    print(f"\n📂 Loading COCO annotations from: {annotation_file}")

    if not annotation_file.exists():
        print(f"❌ Error: Annotation file not found: {annotation_file}")
        print(f"   Please export your CVAT annotations to: {annotation_file}")
        sys.exit(1)

    with open(annotation_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    print(f"✓ Loaded annotations successfully")
    return coco_data

def validate_coco_data(coco_data):
    """验证COCO数据格式"""
    print("\n🔍 Validating COCO data...")

    errors = []

    # 检查必需的字段
    required_fields = ['images', 'annotations', 'categories']
    for field in required_fields:
        if field not in coco_data:
            errors.append(f"Missing required field: {field}")

    if errors:
        print("❌ Validation failed:")
        for error in errors:
            print(f"   - {error}")
        return False

    # 统计信息
    num_images = len(coco_data['images'])
    num_annotations = len(coco_data['annotations'])
    num_categories = len(coco_data['categories'])

    print(f"✓ Validation passed")
    print(f"   - Images: {num_images}")
    print(f"   - Annotations: {num_annotations}")
    print(f"   - Categories: {num_categories}")

    # 检查类别名称是否匹配配置
    category_names = {cat['id']: cat['name'] for cat in coco_data['categories']}
    config_classes = set(config.CLASSES)
    coco_classes = set(category_names.values())

    if config_classes != coco_classes:
        print(f"\n⚠️  Warning: Class mismatch detected!")
        print(f"   Config classes: {config_classes}")
        print(f"   COCO classes: {coco_classes}")
        print(f"   Missing in COCO: {config_classes - coco_classes}")
        print(f"   Extra in COCO: {coco_classes - config_classes}")

    # 每个类别的实例数量
    category_counts = defaultdict(int)
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        cat_name = category_names.get(cat_id, f"Unknown-{cat_id}")
        category_counts[cat_name] += 1

    print(f"\n📊 Instances per category:")
    for cat_name, count in sorted(category_counts.items()):
        print(f"   - {cat_name}: {count}")

    return True

def split_dataset(coco_data, train_ratio, val_ratio, test_ratio, seed=42):
    """划分数据集"""
    print(f"\n✂️  Splitting dataset: Train {train_ratio*100:.0f}% | Val {val_ratio*100:.0f}% | Test {test_ratio*100:.0f}%")

    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 获取所有图像ID
    image_ids = [img['id'] for img in coco_data['images']]
    random.shuffle(image_ids)

    # 计算划分点
    num_images = len(image_ids)
    train_end = int(num_images * train_ratio)
    val_end = train_end + int(num_images * val_ratio)

    # 划分
    train_ids = set(image_ids[:train_end])
    val_ids = set(image_ids[train_end:val_end])
    test_ids = set(image_ids[val_end:])

    print(f"✓ Split complete:")
    print(f"   - Train: {len(train_ids)} images")
    print(f"   - Val: {len(val_ids)} images")
    print(f"   - Test: {len(test_ids)} images")

    return train_ids, val_ids, test_ids

def create_split_coco(coco_data, image_ids, split_name):
    """为特定划分创建COCO数据"""
    split_data = {
        'images': [],
        'annotations': [],
        'categories': coco_data['categories'],
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', [])
    }

    # 筛选图像
    image_id_to_data = {img['id']: img for img in coco_data['images']}
    for img_id in image_ids:
        if img_id in image_id_to_data:
            split_data['images'].append(image_id_to_data[img_id])

    # 筛选标注
    for ann in coco_data['annotations']:
        if ann['image_id'] in image_ids:
            split_data['annotations'].append(ann)

    return split_data

def resize_and_copy_images(coco_data, image_ids, source_dir, target_dir, target_size):
    """调整图片大小并复制到目标目录"""
    print(f"\n🖼️  Processing images for {target_dir.name}...")

    target_dir.mkdir(parents=True, exist_ok=True)

    image_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}

    success_count = 0
    error_count = 0

    for i, img_id in enumerate(image_ids, 1):
        if img_id not in image_id_to_file:
            print(f"⚠️  Warning: Image ID {img_id} not found in annotations")
            continue

        filename = image_id_to_file[img_id]
        source_path = source_dir / filename

        if not source_path.exists():
            print(f"❌ Error: Image not found: {source_path}")
            error_count += 1
            continue

        try:
            # 打开图片
            img = Image.open(source_path)

            # 转换为RGB（处理灰度图和RGBA）
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 保持宽高比缩放
            img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

            # 保存
            target_path = target_dir / filename
            img.save(target_path, quality=95, optimize=True)

            success_count += 1

            if i % 20 == 0:
                print(f"   Processed: {i}/{len(image_ids)}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            error_count += 1

    print(f"✓ Processing complete: {success_count} succeeded, {error_count} failed")

    return success_count, error_count

def generate_dataset_info(coco_data, split_info):
    """生成数据集统计信息"""
    print("\n📊 Generating dataset statistics...")

    info = {
        'total_images': len(coco_data['images']),
        'total_annotations': len(coco_data['annotations']),
        'num_classes': len(coco_data['categories']),
        'classes': [cat['name'] for cat in coco_data['categories']],
        'splits': split_info,
        'config': {
            'image_size': config.IMAGE_SIZE,
            'train_ratio': config.TRAIN_RATIO,
            'val_ratio': config.VAL_RATIO,
            'test_ratio': config.TEST_RATIO,
            'random_seed': config.RANDOM_SEED
        }
    }

    # 每个类别的统计
    category_stats = defaultdict(lambda: {
        'total': 0,
        'train': 0,
        'val': 0,
        'test': 0
    })

    category_names = {cat['id']: cat['name'] for cat in coco_data['categories']}

    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        cat_name = category_names.get(cat_id, f"Unknown-{cat_id}")
        img_id = ann['image_id']

        category_stats[cat_name]['total'] += 1

        if img_id in split_info['train_ids']:
            category_stats[cat_name]['train'] += 1
        elif img_id in split_info['val_ids']:
            category_stats[cat_name]['val'] += 1
        elif img_id in split_info['test_ids']:
            category_stats[cat_name]['test'] += 1

    info['category_stats'] = dict(category_stats)

    return info

def save_dataset_info(info, output_file):
    """保存数据集信息"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"✓ Dataset info saved to: {output_file}")

def print_dataset_summary(info):
    """打印数据集摘要"""
    print("\n" + "="*60)
    print(" Dataset Summary")
    print("="*60)
    print(f"\n📊 Overall Statistics:")
    print(f"   - Total Images: {info['total_images']}")
    print(f"   - Total Annotations: {info['total_annotations']}")
    print(f"   - Number of Classes: {info['num_classes']}")

    print(f"\n📂 Dataset Splits:")
    print(f"   - Train: {info['splits']['train']} images")
    print(f"   - Val: {info['splits']['val']} images")
    print(f"   - Test: {info['splits']['test']} images")

    print(f"\n🏷️  Category Statistics:")
    for cat_name, stats in info['category_stats'].items():
        print(f"   {cat_name}:")
        print(f"      Total: {stats['total']} | Train: {stats['train']} | Val: {stats['val']} | Test: {stats['test']}")

    print("="*60 + "\n")

def main():
    print("\n" + "="*60)
    print(" YOLACT++ Dataset Preparation")
    print("="*60)

    # 创建必要的目录
    config.create_dirs()

    # 1. 加载COCO标注
    coco_data = load_coco_annotations(config.COCO_ANNOTATION_FILE)

    # 2. 验证数据
    if not validate_coco_data(coco_data):
        print("\n❌ Dataset validation failed. Please fix the errors and try again.")
        sys.exit(1)

    # 3. 划分数据集
    train_ids, val_ids, test_ids = split_dataset(
        coco_data,
        config.TRAIN_RATIO,
        config.VAL_RATIO,
        config.TEST_RATIO,
        config.RANDOM_SEED
    )

    # 4. 创建各个划分的COCO文件
    print("\n📝 Creating split annotation files...")

    train_coco = create_split_coco(coco_data, train_ids, 'train')
    val_coco = create_split_coco(coco_data, val_ids, 'val')
    test_coco = create_split_coco(coco_data, test_ids, 'test')

    # 保存划分后的标注文件
    train_ann_file = config.PROCESSED_DATA_DIR / "train" / "annotations.json"
    val_ann_file = config.PROCESSED_DATA_DIR / "val" / "annotations.json"
    test_ann_file = config.PROCESSED_DATA_DIR / "test" / "annotations.json"

    with open(train_ann_file, 'w', encoding='utf-8') as f:
        json.dump(train_coco, f, indent=2, ensure_ascii=False)

    with open(val_ann_file, 'w', encoding='utf-8') as f:
        json.dump(val_coco, f, indent=2, ensure_ascii=False)

    with open(test_ann_file, 'w', encoding='utf-8') as f:
        json.dump(test_coco, f, indent=2, ensure_ascii=False)

    print(f"✓ Annotation files saved")

    # 5. 处理并复制图片
    print("\n🖼️  Processing and copying images...")

    resize_and_copy_images(
        coco_data, train_ids,
        config.RAW_IMAGES_DIR,
        config.PROCESSED_DATA_DIR / "train",
        config.IMAGE_SIZE
    )

    resize_and_copy_images(
        coco_data, val_ids,
        config.RAW_IMAGES_DIR,
        config.PROCESSED_DATA_DIR / "val",
        config.IMAGE_SIZE
    )

    resize_and_copy_images(
        coco_data, test_ids,
        config.RAW_IMAGES_DIR,
        config.PROCESSED_DATA_DIR / "test",
        config.IMAGE_SIZE
    )

    # 6. 生成数据集信息
    split_info = {
        'train': len(train_ids),
        'val': len(val_ids),
        'test': len(test_ids),
        'train_ids': list(train_ids),
        'val_ids': list(val_ids),
        'test_ids': list(test_ids)
    }

    dataset_info = generate_dataset_info(coco_data, split_info)

    # 保存数据集信息
    info_file = config.DATA_ROOT / "dataset_info.json"
    save_dataset_info(dataset_info, info_file)

    # 打印摘要
    print_dataset_summary(dataset_info)

    print("✅ Dataset preparation completed successfully!")
    print(f"\n📁 Processed data location: {config.PROCESSED_DATA_DIR}")
    print(f"📄 Dataset info: {info_file}")
    print("\n🚀 Next step: Run 'python scripts/train.py' to start training")

if __name__ == "__main__":
    main()
