#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
macOS 照片批量处理工具
功能：批量调整照片到目标分辨率
"""

import os
import sys
from pathlib import Path
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from threading import Thread
import datetime

class PhotoResizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("照片批量处理工具")
        self.root.geometry("700x750")
        
        # 设置变量
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.width = tk.IntVar(value=1920)
        self.height = tk.IntVar(value=1080)
        self.keep_aspect = tk.BooleanVar(value=True)
        self.quality = tk.IntVar(value=95)
        self.format_var = tk.StringVar(value="原格式")

        # 批量重命名选项
        self.enable_rename = tk.BooleanVar(value=False)
        self.rename_prefix = tk.StringVar(value="img")
        self.rename_start = tk.IntVar(value=1)
        self.rename_digits = tk.IntVar(value=4)
        
        self.create_widgets()
        
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 输入文件夹选择
        ttk.Label(main_frame, text="输入文件夹:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_folder, width=50).grid(row=0, column=1, pady=5, padx=5)
        ttk.Button(main_frame, text="浏览", command=self.select_input_folder).grid(row=0, column=2, pady=5)
        
        # 输出文件夹选择
        ttk.Label(main_frame, text="输出文件夹:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_folder, width=50).grid(row=1, column=1, pady=5, padx=5)
        ttk.Button(main_frame, text="浏览", command=self.select_output_folder).grid(row=1, column=2, pady=5)
        
        # 分辨率设置
        resolution_frame = ttk.LabelFrame(main_frame, text="分辨率设置", padding="10")
        resolution_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(resolution_frame, text="宽度:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(resolution_frame, textvariable=self.width, width=10).grid(row=0, column=1, pady=5, padx=5)
        ttk.Label(resolution_frame, text="像素").grid(row=0, column=2, sticky=tk.W, pady=5)
        
        ttk.Label(resolution_frame, text="高度:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(resolution_frame, textvariable=self.height, width=10).grid(row=1, column=1, pady=5, padx=5)
        ttk.Label(resolution_frame, text="像素").grid(row=1, column=2, sticky=tk.W, pady=5)
        
        ttk.Checkbutton(resolution_frame, text="保持宽高比", variable=self.keep_aspect).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # 常用分辨率预设
        preset_frame = ttk.Frame(resolution_frame)
        preset_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        ttk.Label(preset_frame, text="预设:").pack(side=tk.LEFT, padx=5)
        presets = [
            ("1920x1080", 1920, 1080),
            ("1280x720", 1280, 720),
            ("3840x2160", 3840, 2160),
            ("1080x1080", 1080, 1080),
            ("800x600", 800, 600)
        ]
        
        for preset_name, w, h in presets:
            ttk.Button(preset_frame, text=preset_name, 
                      command=lambda w=w, h=h: self.set_resolution(w, h)).pack(side=tk.LEFT, padx=2)
        
        # 图片质量设置
        quality_frame = ttk.LabelFrame(main_frame, text="图片质量", padding="10")
        quality_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(quality_frame, text="JPEG质量 (1-100):").grid(row=0, column=0, sticky=tk.W, pady=5)
        quality_scale = ttk.Scale(quality_frame, from_=1, to=100, variable=self.quality, orient=tk.HORIZONTAL, length=300)
        quality_scale.grid(row=0, column=1, pady=5, padx=5)
        ttk.Label(quality_frame, textvariable=self.quality).grid(row=0, column=2, pady=5)
        
        # 输出格式
        ttk.Label(quality_frame, text="输出格式:").grid(row=1, column=0, sticky=tk.W, pady=5)
        format_combo = ttk.Combobox(quality_frame, textvariable=self.format_var,
                                    values=["原格式", "JPEG", "PNG", "WEBP"], state="readonly", width=10)
        format_combo.grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)

        # 批量重命名设置
        rename_frame = ttk.LabelFrame(main_frame, text="批量重命名（数据集命名）", padding="10")
        rename_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        ttk.Checkbutton(rename_frame, text="启用批量重命名", variable=self.enable_rename).grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=5)

        ttk.Label(rename_frame, text="前缀:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(rename_frame, textvariable=self.rename_prefix, width=15).grid(row=1, column=1, pady=5, padx=5)

        ttk.Label(rename_frame, text="起始编号:").grid(row=1, column=2, sticky=tk.W, pady=5, padx=(10,0))
        ttk.Entry(rename_frame, textvariable=self.rename_start, width=10).grid(row=1, column=3, pady=5, padx=5)

        ttk.Label(rename_frame, text="数字位数:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(rename_frame, from_=3, to=6, textvariable=self.rename_digits, width=10).grid(row=2, column=1, pady=5, padx=5)

        ttk.Label(rename_frame, text="示例:").grid(row=2, column=2, sticky=tk.W, pady=5, padx=(10,0))
        self.rename_example = ttk.Label(rename_frame, text="img_0001.jpg", foreground="blue")
        self.rename_example.grid(row=2, column=3, pady=5, padx=5)

        # 更新示例显示
        self.rename_prefix.trace_add('write', self.update_rename_example)
        self.rename_start.trace_add('write', self.update_rename_example)
        self.rename_digits.trace_add('write', self.update_rename_example)

        # 进度条
        self.progress = ttk.Progressbar(main_frame, length=600, mode='determinate')
        self.progress.grid(row=5, column=0, columnspan=3, pady=10)
        
        # 状态标签
        self.status_label = ttk.Label(main_frame, text="准备就绪", foreground="blue")
        self.status_label.grid(row=6, column=0, columnspan=3, pady=5)

        # 日志文本框
        log_frame = ttk.LabelFrame(main_frame, text="处理日志", padding="10")
        log_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        self.log_text = tk.Text(log_frame, height=10, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=8, column=0, columnspan=3, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="开始处理", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="清空日志", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="退出", command=self.root.quit).pack(side=tk.LEFT, padx=5)
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(7, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
    def select_input_folder(self):
        folder = filedialog.askdirectory(title="选择输入文件夹")
        if folder:
            self.input_folder.set(folder)
            self.log(f"选择输入文件夹: {folder}")
            
    def select_output_folder(self):
        folder = filedialog.askdirectory(title="选择输出文件夹")
        if folder:
            self.output_folder.set(folder)
            self.log(f"选择输出文件夹: {folder}")
            
    def set_resolution(self, width, height):
        self.width.set(width)
        self.height.set(height)
        self.log(f"设置分辨率为: {width}x{height}")

    def update_rename_example(self, *args):
        """更新重命名示例"""
        try:
            prefix = self.rename_prefix.get()
            start = self.rename_start.get()
            digits = self.rename_digits.get()
            example = f"{prefix}_{start:0{digits}d}.jpg"
            self.rename_example.config(text=example)
        except:
            self.rename_example.config(text="img_0001.jpg")

    def log(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.update()
        
    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
        
    def start_processing(self):
        # 验证输入
        if not self.input_folder.get():
            messagebox.showerror("错误", "请选择输入文件夹")
            return
            
        if not self.output_folder.get():
            messagebox.showerror("错误", "请选择输出文件夹")
            return
            
        if self.width.get() <= 0 or self.height.get() <= 0:
            messagebox.showerror("错误", "请输入有效的分辨率")
            return
        
        # 在新线程中处理，避免界面冻结
        self.start_button.config(state=tk.DISABLED)
        thread = Thread(target=self.process_images)
        thread.daemon = True
        thread.start()
        
    def process_images(self):
        input_dir = Path(self.input_folder.get())
        output_dir = Path(self.output_folder.get())
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 支持的图片格式
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
        
        # 获取所有图片文件
        image_files = [f for f in input_dir.rglob('*') if f.suffix.lower() in supported_formats]
        
        if not image_files:
            self.log("错误: 未找到图片文件")
            self.status_label.config(text="未找到图片文件", foreground="red")
            self.start_button.config(state=tk.NORMAL)
            return
        
        total_files = len(image_files)
        self.log(f"找到 {total_files} 个图片文件")
        self.progress['maximum'] = total_files
        
        success_count = 0
        error_count = 0
        
        for idx, image_path in enumerate(image_files, 1):
            try:
                # 更新进度
                self.progress['value'] = idx
                self.status_label.config(text=f"处理中: {idx}/{total_files}", foreground="blue")
                
                # 打开图片
                img = Image.open(image_path)
                original_size = img.size
                
                # 计算新尺寸
                target_width = self.width.get()
                target_height = self.height.get()
                
                if self.keep_aspect.get():
                    # 保持宽高比
                    img.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
                    new_size = img.size
                else:
                    # 强制缩放到指定尺寸
                    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    new_size = (target_width, target_height)
                
                # 确定输出格式和文件名
                output_format = self.format_var.get()
                if output_format == "原格式":
                    output_ext = image_path.suffix
                    save_format = img.format if img.format else "JPEG"
                elif output_format == "JPEG":
                    output_ext = ".jpg"
                    save_format = "JPEG"
                    # 转换 RGBA 到 RGB
                    if img.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                        img = background
                elif output_format == "PNG":
                    output_ext = ".png"
                    save_format = "PNG"
                elif output_format == "WEBP":
                    output_ext = ".webp"
                    save_format = "WEBP"
                
                # 生成输出文件名
                if self.enable_rename.get():
                    # 使用批量重命名
                    prefix = self.rename_prefix.get()
                    start_num = self.rename_start.get()
                    digits = self.rename_digits.get()
                    file_number = start_num + idx - 1
                    output_filename = f"{prefix}_{file_number:0{digits}d}{output_ext}"
                    output_path = output_dir / output_filename
                else:
                    # 保持原文件名
                    output_filename = image_path.stem + output_ext
                    output_path = output_dir / output_filename

                    # 如果文件存在，添加数字后缀
                    counter = 1
                    while output_path.exists():
                        output_filename = f"{image_path.stem}_{counter}{output_ext}"
                        output_path = output_dir / output_filename
                        counter += 1
                
                # 保存图片
                save_kwargs = {}
                if save_format == "JPEG":
                    save_kwargs['quality'] = self.quality.get()
                    save_kwargs['optimize'] = True
                elif save_format == "WEBP":
                    save_kwargs['quality'] = self.quality.get()
                elif save_format == "PNG":
                    save_kwargs['optimize'] = True
                
                img.save(output_path, format=save_format, **save_kwargs)
                
                self.log(f"✓ {image_path.name}: {original_size} → {new_size}")
                success_count += 1
                
            except Exception as e:
                self.log(f"✗ {image_path.name}: 错误 - {str(e)}")
                error_count += 1
        
        # 完成
        self.progress['value'] = total_files
        self.status_label.config(text="处理完成!", foreground="green")
        self.log(f"\n处理完成! 成功: {success_count}, 失败: {error_count}")
        self.start_button.config(state=tk.NORMAL)
        
        messagebox.showinfo("完成", f"处理完成!\n成功: {success_count}\n失败: {error_count}")


def main():
    # 检查是否安装了 Pillow
    try:
        from PIL import Image
    except ImportError:
        print("错误: 未安装 Pillow 库")
        print("请运行: pip install Pillow")
        sys.exit(1)
    
    root = tk.Tk()
    app = PhotoResizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
