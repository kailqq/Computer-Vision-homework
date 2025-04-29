import os
import sys
import numpy as np
from scipy.linalg import eigh
import cv2
import pickle
from utils import preprocess_image, create_feature_faces_image

def train_model(data_dir, energy_percentage, model_dir):
    """训练模型"""
    if os.path.isdir(model_dir):
        model_path = os.path.join(model_dir, 'model.pkl')
    else:
        model_path = model_dir
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.pgm', '.jpg', '.png'))]
    
    # 预处理所有图像
    images = []
    for img_file in image_files:
        img_path = os.path.join(data_dir, img_file)
        try:
            processed = preprocess_image(img_path)
            images.append(processed)
        except Exception as e:
            print(f"处理图像 {img_file} 时出错: {e}")
            continue
    
    if not images:
        raise ValueError("没有有效的训练图像")
    
    # 将图像转换为矩阵
    X = np.array(images)
    n_samples, height, width = X.shape
    X = X.reshape(n_samples, -1)  # 展平为2D矩阵
    
    # 计算平均脸
    mean_face = np.mean(X, axis=0)
    X_centered = X - mean_face
    
    # 计算协方差矩阵
    cov_matrix = np.cov(X_centered.T)
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = eigh(cov_matrix)
    
    # 按特征值大小排序
    idx = np.argsort(eigenvalues)[::-1] #这里默认是从小到大，需要reverse一下
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 根据能量百分比选择特征脸数量
    total_energy = np.sum(eigenvalues)
    cumulative_energy = np.cumsum(eigenvalues)
    n_components = np.argmax(cumulative_energy >= energy_percentage * total_energy) + 1
   
    # 选择前n_components个特征向量
    eigenfaces = eigenvectors[:, :n_components]
    
    # 计算所有训练图像在特征空间中的投影
    train_projections = np.dot(X_centered, eigenfaces) # X_centered (n, HxW)  eigenfaces(HxW,d)
                                                       # list(n,d)
    # 保存模型
    model = {
        'mean_face': mean_face,
        'eigenfaces': eigenfaces,
        'train_projections': train_projections,
        'image_files': image_files,
        'image_size': (height, width)
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    feature_faces_image = create_feature_faces_image(eigenfaces.T, (height, width))
    cv2.putText(feature_faces_image, f"Eigenfaces: {n_components}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Eigenfaces', feature_faces_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return model

def main():
    if len(sys.argv) < 4:
        print("用法: python train.py <数据目录> <能量百分比> <模型文件>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    energy_percentage = float(sys.argv[2])
    model_dir = sys.argv[3]

    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 {data_dir} 不存在")
        sys.exit(1)
    
    try:
        model = train_model(data_dir, energy_percentage, model_dir)
        print(f"模型训练完成，保存到 {model_dir}")
        print(f"使用的特征脸数量: {model['eigenfaces'].shape[1]}")
    except Exception as e:
        print(f"训练过程中出错: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()