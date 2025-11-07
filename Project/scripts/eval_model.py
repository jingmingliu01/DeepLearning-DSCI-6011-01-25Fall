#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLACT++ Model Evaluation Script
在测试集上评估模型性能
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

def find_best_model():
    """查找最佳模型权重"""
    if config.BEST_WEIGHTS.exists():
        return config.BEST_WEIGHTS

    # 查找最新的checkpoint
    checkpoints = list(config.CHECKPOINTS_DIR.glob("*.pth"))
    if not checkpoints:
        print(f"❌ Error: No model weights found in {config.CHECKPOINTS_DIR}")
        return None

    # 按修改时间排序
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"ℹ️  Using latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

def run_evaluation(model_path, output_dir=None):
    """运行评估"""
    if output_dir is None:
        output_dir = config.RESULTS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n📊 Starting model evaluation...")
    print(f"\n评估配置:")
    print(f"  - Model: {model_path}")
    print(f"  - Test Data: {config.PROCESSED_DATA_DIR / 'test'}")
    print(f"  - Output: {output_dir}")

    cmd = [
        'python', str(config.YOLACT_ROOT / 'eval.py'),
        f'--trained_model={model_path}',
        f'--score_threshold={config.INFERENCE_SCORE_THRESHOLD}',
        f'--top_k={config.TOP_K}',
        f'--config=campus_objects_config',
        f'--output_coco_json',
        '--dataset=campus_objects_dataset:test',
    ]

    if config.USE_GPU:
        cmd.append('--cuda=True')

    print(f"\n执行命令:")
    print(f"  {' '.join(cmd)}")
    print("\n" + "="*60)

    try:
        os.chdir(config.YOLACT_ROOT)
        subprocess.run(cmd, check=True)

        print("\n" + "="*60)
        print("✅ Evaluation completed successfully!")
        print(f"\n结果保存在: {output_dir}")

        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Evaluation failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        return False
    finally:
        os.chdir(PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLACT++ model')
    parser.add_argument('--model', type=str,
                        help='Path to model weights (default: auto-detect)')
    parser.add_argument('--output', type=str,
                        help='Output directory for results')
    args = parser.parse_args()

    print("\n" + "="*60)
    print(" YOLACT++ Model Evaluation")
    print("="*60)

    # 查找模型
    model_path = args.model if args.model else find_best_model()
    if model_path is None:
        sys.exit(1)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Error: Model not found: {model_path}")
        sys.exit(1)

    # 运行评估
    output_dir = Path(args.output) if args.output else None
    success = run_evaluation(model_path, output_dir)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
