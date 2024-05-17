import numpy as np


"""
    本py文件提供对图像进行pca白化处理的函数pca_whitening和进行
    zca白化处理的函数zca_whitening，
    其输入接口为matplotlib.image库的imread函数对图像的读取结果。
    输出同上。
"""

def normalize_image(image):
    # 将图像数据归一化
    image_min = image.min()
    image_max = image.max()
    return ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)

def pca_whitening(image):
    original_shape = image.shape  # 保存原始形状
    X = image.reshape(-1, 3)

    # 减去平均值以中心化数据
    X_centered = X - np.mean(X, axis=0)

    # 计算协方差矩阵
    cov = np.cov(X_centered, rowvar=False)

    # 计算协方差矩阵的特征值和特征向量
    U, S, Ut = np.linalg.svd(cov)

    # 计算白化矩阵
    epsilon = 1e-5  # 防止除以零
    D = np.diag(1.0 / np.sqrt(S + epsilon))

    # 白化变换
    X_whitened = X_centered.dot(U).dot(D)

    # 将白化后的数据重塑回原始图像的形状
    return normalize_image(X_whitened.reshape(original_shape))

def zca_whitening(image):
    original_shape = image.shape  # 保存原始形状
    X = image.reshape(-1, 3)

    # 减去平均值以中心化数据
    X_centered = X - np.mean(X, axis=0)

    # 计算协方差矩阵
    cov = np.cov(X_centered, rowvar=False)

    # 计算协方差矩阵的特征值和特征向量
    U, S, Vt = np.linalg.svd(cov)

    # 计算白化矩阵
    epsilon = 1e-5  # 防止除以零
    D = np.diag(1.0 / np.sqrt(S + epsilon))

    # 白化变换
    X_whitened = X_centered.dot(U).dot(D)
    
    # ZCA白化变换
    X_zca_whitened = X_whitened.dot(U.T)

    # 将白化后的数据重塑回原始图像的形状
    return normalize_image(X_zca_whitened.reshape(original_shape))
