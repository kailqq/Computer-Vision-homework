import cv2
import numpy as np
import os
import sys
import glob
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import logging
def cv2_add_chinese_text(img, text, position, font_path, font_size, color):
    """使用PIL绘制中文字符
       img: 输入图像
       text: 要绘制的文本
       position: 文本位置(x, y)
       font_path: 字体路径
       font_size: 字体大小
       color: 文本颜色(BGR格式)

       因为OpenCV对中文的支持并不友好
    """
    # 确保图像是uint8类型
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def create_intro(width, height, duration_sec=3, fps=30, font_path="simhei.ttf"):
    """创建片头
    width: 视频宽度
    height: 视频高度
    duration_sec: 片头持续时间
    fps: 视频帧率
    font_path: 字体路径
    """
    frames = []
    total_frames = duration_sec * fps
    for i in range(total_frames):
        # 创建渐变背景
        progress = i / total_frames
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        frame = (frame * progress).astype(np.uint8)
        # 添加文本
        text = ["个人视频作品","素材来源于网络"]
        font_size = 72
        text_x = width // 2 - len(text) * font_size // 3
        text_y = height // 2 - font_size // 2
        color = (0, 0, int(255 * (1-progress)))
        if progress < 0.3:
            frame = cv2_add_chinese_text(frame, text[0], (text_x, text_y), font_path, font_size, color)
        else:
            frame = cv2_add_chinese_text(frame, text[1], (text_x, text_y), font_path, font_size, color)
        frames.append(frame)
    
    return frames

def resize_image(image, target_width, target_height):
    """调整图像大小保持宽高比
    image: 输入图像
    target_width: 目标宽度
    target_height: 目标高度

    与resize不同，这里会保持宽高比，通过选择较小的比例来实现
    """
    h, w = image.shape[:2]
    ratio = min(target_width/w, target_height/h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized = cv2.resize(image, (new_w, new_h))
    # 创建画布并将图像放在中央
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def add_subtitle(frame, text, font_path="simhei.ttf"):
    """添加字幕到视频帧底部
    frame: 输入视频帧
    text: 要绘制的文本
    font_path: 字体路径
    """
    h, w = frame.shape[:2]  
    # 添加半透明背景条
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-60), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    # 添加中文文字
    font_size = 28
    text_x = w // 2 - len(text) * font_size // 4
    text_y = h - 45
    
    return cv2_add_chinese_text(frame, text, (text_x, text_y), font_path, font_size, (255, 255, 255))

def create_transition(frame1, frame2, num_frames=15):
    """创建两帧之间的过渡效果
    frame1: 输入视频帧1
    frame2: 输入视频帧2
    num_frames: 过渡帧数

    过渡效果所用函数可见https://docs.opencv.org/3.4/d5/dc4/tutorial_adding_images.html
    """
    transitions = []
    for i in range(num_frames):
        alpha = i / num_frames
        transition = cv2.addWeighted(frame1, 1-alpha, frame2, alpha, 0)
        transitions.append(transition)
    return transitions

def main():

    info ={"name":"张晋恺","id":"3230102400"}
    if len(sys.argv) != 2:
        print("用法: python script.py <输入文件夹路径>")
        print("例如: python script.py C:\\input")
        sys.exit(1)
    
    input_dir = sys.argv[1]


    font_paths = [
        "simhei.ttf",  # 当前目录
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux常见位置
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # 备选字体
    ]
    
    font_path = None
    for path in font_paths:
        if os.path.exists(path):
            font_path = path
            print(f"使用字体: {font_path}")
            break
    
    if font_path is None:
        print("警告: 找不到可用的字体文件")
        sys.exit(1)
    video_files = glob.glob(os.path.join(input_dir, "*.avi")) + glob.glob(os.path.join(input_dir, "*.webm")) + glob.glob(os.path.join(input_dir, "*.mp4"))
    image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.jpeg")) + glob.glob(os.path.join(input_dir, "*.png"))
    
    if not video_files:
        print("错误: 未找到视频文件")
        sys.exit(1)
    
    if len(image_files) < 5:
        print("错误: 需要至少5张图片")
        sys.exit(1)
    
    # 打开原始视频以获取尺寸和帧率
    video_path = video_files[0]
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        sys.exit(1)
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 创建输出视频
    output_path = os.path.join(os.path.dirname(input_dir), "MyVideo_Output.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 1. 创建片头
    intro_frames = create_intro(width, height, 5, fps, font_path)
    for frame in intro_frames:
        frame_with_subtitle = add_subtitle(frame.copy(), f"学号: {info['id']} 姓名: {info['name']}", font_path)
        out.write(frame_with_subtitle)
    
    # 2. 处理图片为幻灯片
    photo_duration_sec = 2  # 每张照片显示的秒数
    frames_per_photo = photo_duration_sec * fps
    transition_frames = 15  # 过渡帧数
    
    prev_frame = None
    for img_path in image_files:
        # 读取和调整图像大小
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图像 {img_path}")
            continue
       
        resized_img = resize_image(img, width, height)
        frame_with_subtitle = add_subtitle(resized_img.copy(), f"学号: {info['id']} 姓名: {info['name']}", font_path)
        if prev_frame is not None:
            transitions = create_transition(prev_frame, frame_with_subtitle, transition_frames)
            for frame in transitions:
                out.write(frame)
        for _ in range(int(frames_per_photo - transition_frames)):
            out.write(frame_with_subtitle)
        
        prev_frame = frame_with_subtitle
    
    # 3. 添加原始视频
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到视频开始
    
    # 添加从图片到视频的过渡
    if prev_frame is not None:
        ret, video_frame = cap.read()
        if ret:
            video_frame_with_subtitle = add_subtitle(video_frame.copy(), f"学号: {info['id']} 姓名: {info['name']}", font_path)
            transitions = create_transition(prev_frame, video_frame_with_subtitle, transition_frames)
            for frame in transitions:
                out.write(frame)

    # 处理视频的其余部分
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到视频开始
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_with_subtitle = add_subtitle(frame.copy(), f"学号: {info['id']} 姓名: {info['name']}", font_path)
        out.write(frame_with_subtitle)
    cap.release()
    out.release()
    
    print(f"视频已成功创建: {output_path}")
if __name__ == "__main__":
    main()