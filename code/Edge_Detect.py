import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from skimage import filters, measure, morphology
from Fliter import gaussian_filter

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


def canny_detect_edges(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算图像的中值
    v = np.median(gray)

    # 设置Canny边缘检测的上下阈值
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    # 使用Canny边缘检测算法
    edges = cv2.Canny(gray, lower, upper)

    # 在原图上绘制边缘
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255, 0, 0), 2)

    # 将 BGR 图像转换为 RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return rgb_image