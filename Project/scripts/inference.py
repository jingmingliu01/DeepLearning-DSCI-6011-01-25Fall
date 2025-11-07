#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLACT++ Inference Script
对单张图片或图片目录进行推理
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

    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"ℹ️  Using latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

def run_inference_image(model_path, image_path, output_path=None, display=True):
    """对单张图片进行推理"""
    if output_path is None:
        output_path = config.RESULTS_DIR / "images" / f"result_{Path(image_path).name}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n🖼️  Running inference on: {image_path}")

    cmd = [
        'python', str(config.YOLACT_ROOT / 'eval.py'),
        f'--trained_model={model_path}',
        f'--score_threshold={config.INFERENCE_SCORE_THRESHOLD}',
        f'--top_k={config.TOP_K}',
        f'--config=campus_objects_config',
        f'--image={image_path}:{output_path}',
    ]

    if config.USE_GPU:
        cmd.append('--cuda=True')

    if display:
        cmd.append('--display')

    try:
        os.chdir(config.YOLACT_ROOT)
        subprocess.run(cmd, check=True)

        print(f"✅ Inference completed!")
        print(f"   Result saved to: {output_path}")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Inference failed: {e}")
        return False
    finally:
        os.chdir(PROJECT_ROOT)

def run_inference_folder(model_path, input_folder, output_folder=None):
    """对文件夹中的所有图片进行推理"""
    if output_folder is None:
        output_folder = config.RESULTS_DIR / "images"

    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Running inference on folder: {input_folder}")

    cmd = [
        'python', str(config.YOLACT_ROOT / 'eval.py'),
        f'--trained_model={model_path}',
        f'--score_threshold={config.INFERENCE_SCORE_THRESHOLD}',
        f'--top_k={config.TOP_K}',
        f'--config=campus_objects_config',
        f'--images={input_folder}:{output_folder}',
    ]

    if config.USE_GPU:
        cmd.append('--cuda=True')

    try:
        os.chdir(config.YOLACT_ROOT)
        subprocess.run(cmd, check=True)

        print(f"✅ Batch inference completed!")
        print(f"   Results saved to: {output_folder}")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Batch inference failed: {e}")
        return False
    finally:
        os.chdir(PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description='Run YOLACT++ inference')
    parser.add_argument('--model', type=str,
                        help='Path to model weights (default: auto-detect)')
    parser.add_argument('--image', type=str,
                        help='Path to input image')
    parser.add_argument('--folder', type=str,
                        help='Path to input folder')
    parser.add_argument('--output', type=str,
                        help='Path to output file/folder')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display result')
    args = parser.parse_args()

    print("\n" + "="*60)
    print(" YOLACT++ Inference")
    print("="*60)

    # 检查输入
    if not args.image and not args.folder:
        print("❌ Error: Must specify either --image or --folder")
        parser.print_help()
        sys.exit(1)

    # 查找模型
    model_path = args.model if args.model else find_best_model()
    if model_path is None:
        sys.exit(1)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Error: Model not found: {model_path}")
        sys.exit(1)

    print(f"\n📦 Model: {model_path}")

    # 运行推理
    success = False
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"❌ Error: Image not found: {image_path}")
            sys.exit(1)

        output_path = Path(args.output) if args.output else None
        success = run_inference_image(
            model_path, image_path, output_path,
            display=not args.no_display
        )

    elif args.folder:
        input_folder = Path(args.folder)
        if not input_folder.exists():
            print(f"❌ Error: Folder not found: {input_folder}")
            sys.exit(1)

        output_folder = Path(args.output) if args.output else None
        success = run_inference_folder(model_path, input_folder, output_folder)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
