import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from scipy import fftpack
from skimage import filters, measure, morphology

def high_pass_filter_and_binary(image):

    # 进行傅里叶变换
    f = fft2(image)

    # 创建高通滤波器
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    fshift = np.fft.fftshift(f)
    fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0  # 阻止中心区域（低频）

    # 逆傅里叶变换
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.abs(ifft2(f_ishift))

    # 对高通滤波后的图像进行二值化
    threshold = filters.threshold_otsu(img_back)  # 使用Otsu方法自动选择阈值
    binary_img = img_back > threshold

    return binary_img

def low_pass_filter_and_binary(image):
    # 进行傅里叶变换
    f = fftpack.fft2(image)
    
    # 创建低通滤波器
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    fshift = fftpack.fftshift(f)
    
    # 创建一个与图像大小相同的全0数组
    mask = np.zeros((rows, cols), dtype=bool)
    
    # 将中心区域设为1（低频部分）
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
    
    # 应用低通滤波器
    fshift *= mask
    
    # 逆傅里叶变换
    f_ishift = fftpack.ifftshift(fshift)
    img_back = np.abs(fftpack.ifft2(f_ishift))
    
    # 对低通滤波后的图像进行二值化
    threshold = filters.threshold_otsu(img_back)  # 使用Otsu方法自动选择阈值
    binary_img = img_back > threshold
    
    return binary_img

def band_pass_filter_and_binary(image):
    # 进行傅里叶变换
    f = fftpack.fft2(image)
    
    # 创建带通滤波器
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    fshift = fftpack.fftshift(f)
    
    # 创建一个带通滤波器，这里以D0为频率中心，宽度为W的矩形窗为例
    D0 = 30
    W = 10
    band_pass = np.zeros((rows, cols))
    band_pass[crow - D0 - W:crow - D0 + W, ccol - D0 - W:ccol - D0 + W] = 1
    band_pass[crow + D0 - W:crow + D0 + W, ccol + D0 - W:ccol + D0 + W] = 1
    
    fshift = fshift * band_pass  # 应用带通滤波器
    
    # 逆傅里叶变换
    f_ishift = fftpack.ifftshift(fshift)
    img_back = np.abs(fftpack.ifft2(f_ishift))
    
    # 对带通滤波后的图像进行二值化
    threshold = filters.threshold_otsu(img_back)  # 使用Otsu方法自动选择阈值
    binary_img = img_back > threshold
    
    return binary_img

def band_reject_filter_and_binary(image):
    # 进行傅里叶变换
    f = fftpack.fft2(image)
    
    # 创建带阻滤波器
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    fshift = fftpack.fftshift(f)
    
    # 创建一个带阻滤波器，这里以D0为频率中心，宽度为W的矩形窗为例
    D0 = 30
    W = 10
    band_reject = np.ones((rows, cols))
    band_reject[crow - D0 - W:crow - D0 + W, ccol - D0 - W:ccol - D0 + W] = 0
    band_reject[crow + D0 - W:crow + D0 + W, ccol + D0 - W:ccol + D0 + W] = 0
    
    fshift = fshift * band_reject  # 应用带阻滤波器
    
    # 逆傅里叶变换
    f_ishift = fftpack.ifftshift(fshift)
    img_back = np.abs(fftpack.ifft2(f_ishift))
    
    # 对带阻滤波后的图像进行二值化
    threshold = filters.threshold_otsu(img_back)  # 使用Otsu方法自动选择阈值
    binary_img = img_back > threshold
    
    return binary_img
