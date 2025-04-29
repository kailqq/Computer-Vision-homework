import os
import sys
import numpy as np
import cv2
import pickle
from utils import preprocess_image

def load_model(model_path):
    """加载训练好的模型"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def recognize_face(test_image_path, model_path):
    """识别人脸,将测试文件与训练数据在特征空间进行比对，返回nearest neighbor"""
    # 加载模型
    model = load_model(model_path)
    # 预处理测试图像
    test_image = preprocess_image(test_image_path)
    test_image = test_image.reshape(-1)
    # 计算测试图像在特征脸空间中的投影
    test_image_centered = test_image - model['mean_face']
    test_projection = np.dot(test_image_centered, model['eigenfaces']) # (d,)
    train_projections = model['train_projections'] # list n of (d,)
    # 计算与所有训练图像的欧氏距离
    distances = [np.linalg.norm(test_projection - proj) for proj in train_projections] # linalg 函数可以计算两个向量之间的欧氏距离
    # 找到最相似的图像
    min_distance_idx = np.argmin(distances)
    min_distance = distances[min_distance_idx]
    most_similar_image = model['image_files'][min_distance_idx]
    return most_similar_image, min_distance

def main():
    if len(sys.argv) < 4:
        print("用法: python test.py <测试图像> <模型文件> <训练数据目录>")
        sys.exit(1)
    test_image_path = sys.argv[1]
    model_path = sys.argv[2]
    data_dir = sys.argv[3]
    if not os.path.exists(test_image_path):
        print(f"错误: 测试图像 {test_image_path} 不存在")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        sys.exit(1)
    try:
        most_similar_image, distance = recognize_face(test_image_path, model_path)
        test_img = cv2.imread(test_image_path)
        similar_img = cv2.imread(os.path.join(data_dir, most_similar_image))
        cv2.putText(test_img, f"Distance: {distance:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Test Image', test_img)
        cv2.imshow('Most Similar Image', similar_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()        
        print(f"最相似的图像: {most_similar_image}")
        print(f"距离: {distance}")
        
    except Exception as e:
        print(f"识别过程中出错: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()