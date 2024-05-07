import cv2
import numpy as np
import matplotlib.pyplot as plt

# 高斯滤波
def gaussian_filter(image, filter_size, sigma): # 参数：图像对应矩阵，卷积核大小，方差
    # 确保滤波器大小是奇数
    if filter_size % 2 == 0:
        filter_size += 1

    # 获取图像的高度和宽度
    height, width = image.shape
    
    # 创建一个新的图像用于存储滤波后的结果
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    
    # 计算高斯核
    kernel = np.zeros((filter_size, filter_size), dtype=np.float32)
    for i in range(-filter_size // 2, filter_size // 2 + 1):
        for j in range(-filter_size // 2, filter_size // 2 + 1):
            distance = np.sqrt((i * i) + (j * j))
            weight = np.exp(-(distance ** 2) / (2 * sigma ** 2))
            kernel[i + filter_size // 2][j + filter_size // 2] = weight
    kernel /= np.sum(kernel)  # 归一化
    
    # 应用高斯滤波
    for i in range(height):
        for j in range(width):
            window = np.zeros((filter_size, filter_size), dtype=np.float32)
            for m in range(-filter_size // 2, filter_size // 2 + 1):
                for n in range(-filter_size // 2, filter_size // 2 + 1):
                    if 0 <= i + m < height and 0 <= j + n < width:
                        window[m + filter_size // 2][n + filter_size // 2] = image[i + m, j + n] * kernel[m + filter_size // 2][n + filter_size // 2]
            
            filtered_value = np.sum(window)
            filtered_image[i, j] = np.uint8(filtered_value)
    
    return filtered_image

# 均值滤波
def mean_filter(image, filter_size):
    # 确保滤波器大小是奇数
    if filter_size % 2 == 0:
        filter_size += 1

    height, width = image.shape
    
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            # 定义一个窗口用于存储邻域像素
            window = np.zeros((filter_size, filter_size), dtype=np.uint8)
            
            # 填充窗口
            for m in range(-filter_size // 2, filter_size // 2 + 1):
                for n in range(-filter_size // 2, filter_size // 2 + 1):
                    if 0 <= i + m < height and 0 <= j + n < width:
                        window[m + filter_size // 2][n + filter_size // 2] = image[i + m, j + n]
            
            mean_value = np.mean(window)
            filtered_image[i, j] = mean_value
    
    return filtered_image

# 中值滤波
def median_filter(image, filter_size):
    # 确保滤波器大小是奇数
    if filter_size % 2 == 0:
        filter_size += 1
    
    # 获取图像的尺寸
    height, width = image.shape
    
    # 创建一个与原图大小相同的空白图像
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            # 定义局部区域的边界
            min_i = max(i - filter_size // 2, 0)
            max_i = min(i + filter_size // 2, height - 1)
            min_j = max(j - filter_size // 2, 0)
            max_j = min(j + filter_size // 2, width - 1)
            
            region = image[min_i:max_i + 1, min_j:max_j + 1]
            median = np.median(region)
            filtered_image[i, j] = median
    
    return filtered_image

