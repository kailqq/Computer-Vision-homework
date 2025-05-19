import cv2
import numpy as np
import os

def load_eye_positions(image_path):
    """加载眼睛位置信息"""
    txt_path = os.path.splitext(image_path)[0] + '.eye'
    if not os.path.exists(txt_path):
        return None
    with open(txt_path, 'r') as f:
        # 跳过第一行（标题行）
        f.readline()
        # 读取第二行的眼睛坐标
        positions = list(map(float, f.readline().strip().split()))
    return positions
def align_face(image, eye_positions):
    """根据眼睛位置对齐人脸"""
    # 计算眼睛连线的角度
    left_eye = np.array(eye_positions[:2])
    right_eye = np.array(eye_positions[2:])
    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], left_eye[0] - right_eye[0])) # 现在人眼的角度，由于镜像，靠左边的是右眼，右边的是左眼
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)#逆时针旋转
    aligned = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return aligned

def preprocess_image(image_path, target_size=(100, 100)):
    """预处理图像：读取、对齐、调整大小、转换为灰度图"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eye_positions = load_eye_positions(image_path)
    if eye_positions is None:
        print(f"未找到眼睛位置信息: {image_path},不进行对齐")
        resized = cv2.resize(gray, target_size)
        normalized = resized.astype(np.float32) / 255.0
    else:
        aligned = align_face(gray, eye_positions)
        resized = cv2.resize(aligned, target_size)
        normalized = resized.astype(np.float32) / 255.0
    return normalized

def create_feature_faces_image(eigenfaces, image_size=(100, 100)):
    """创建特征脸图像"""
    # 取前10个特征脸
    num_faces = min(10, eigenfaces.shape[0])
    faces = eigenfaces[:num_faces]
    # 归一化特征脸,0-1
    faces = (faces - faces.min()) / (faces.max() - faces.min())
    # 调整大小并拼接
    rows = 2
    cols = 5
    result = np.zeros((rows * image_size[0], cols * image_size[1]))
    for i in range(num_faces):
        face = faces[i].reshape(image_size)
        row = i // cols
        col = i % cols
        result[row*image_size[0]:(row+1)*image_size[0], 
               col*image_size[1]:(col+1)*image_size[1]] = face
    return result