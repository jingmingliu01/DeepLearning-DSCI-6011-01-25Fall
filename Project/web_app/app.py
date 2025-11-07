#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLACT++ Web Application
Flask-based web interface for campus object detection
"""

import sys
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import subprocess
import uuid
import time

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER

# 创建上传目录
config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def find_model():
    """查找模型权重"""
    if config.BEST_WEIGHTS.exists():
        return config.BEST_WEIGHTS

    checkpoints = list(config.CHECKPOINTS_DIR.glob("*.pth"))
    if not checkpoints:
        return None

    return max(checkpoints, key=lambda p: p.stat().st_mtime)

def run_yolact_inference(model_path, image_path, output_path):
    """运行YOLACT推理"""
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

    try:
        os.chdir(config.YOLACT_ROOT)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running inference: {e}")
        return False
    finally:
        os.chdir(PROJECT_ROOT)

@app.route('/')
def index():
    """主页"""
    return render_template('index.html',
                          classes=config.CLASSES,
                          num_classes=config.NUM_CLASSES)

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和推理"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # 生成唯一文件名
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}.{ext}"
        filepath = config.UPLOAD_FOLDER / filename

        # 保存上传的文件
        file.save(filepath)

        # 查找模型
        model_path = find_model()
        if model_path is None:
            return jsonify({'error': 'Model not found'}), 500

        # 运行推理
        output_filename = f"{unique_id}_result.{ext}"
        output_path = config.UPLOAD_FOLDER / output_filename

        start_time = time.time()
        success = run_yolact_inference(model_path, filepath, output_path)
        inference_time = time.time() - start_time

        if success and output_path.exists():
            return jsonify({
                'success': True,
                'original_image': f'/uploads/{filename}',
                'result_image': f'/uploads/{output_filename}',
                'inference_time': f'{inference_time:.2f}s'
            })
        else:
            return jsonify({'error': 'Inference failed'}), 500

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """获取上传的文件"""
    return send_from_directory(config.UPLOAD_FOLDER, filename)

@app.route('/about')
def about():
    """关于页面"""
    return render_template('about.html')

if __name__ == '__main__':
    # 检查模型是否存在
    model_path = find_model()
    if model_path is None:
        print("⚠️  Warning: No trained model found!")
        print(f"   Please train the model first or place model weights in: {config.WEIGHTS_DIR}")
        print(f"   The app will start but inference will not work.\n")
    else:
        print(f"✓ Model loaded: {model_path}\n")

    print(f"🚀 Starting Flask app on http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    print(f"   Press Ctrl+C to stop\n")

    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )
