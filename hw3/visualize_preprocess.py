import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import preprocess_image, load_eye_positions, align_face

def visualize_preprocess(image_path):
    """可视化图像处理的每个步骤"""
    # 读取原始图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 加载眼睛位置
    eye_positions = load_eye_positions(image_path)
    
    # 对齐人脸
    aligned = align_face(gray, eye_positions)
    
    # 调整大小
    resized = cv2.resize(aligned, (100, 100))
    print(resized[0])
    # 归一化
    normalized = resized.astype(np.float32) / 255.0
    print(normalized[0])
    
    # 创建子图
    plt.figure(figsize=(15, 5))
    
    # 原始图像
    plt.subplot(151)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('original image')
    plt.axis('off')
    
    # 灰度图
    plt.subplot(152)
    plt.imshow(gray, cmap='gray')
    plt.title('gray image')
    plt.axis('off')
    
    # 对齐后
    plt.subplot(153)
    plt.imshow(aligned, cmap='gray')
    plt.title('aligned image')
    plt.axis('off')
    
    # 调整大小后
    plt.subplot(154)
    plt.imshow(resized, cmap='gray')
    plt.title('resized image')
    plt.axis('off')
    
    # 归一化后
    plt.subplot(155)
    plt.imshow(normalized, cmap='gray')
    plt.title('normalized image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 测试图像路径
    test_image_path = "./data/dataset2/image.png"  # 请替换为实际的图像路径
    visualize_preprocess(test_image_path) 