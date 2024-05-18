import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from skimage import filters, measure, morphology
from Filter import gaussian_filter

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

    # 计算灰度图像的傅里叶变换
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    # 显示频域的幅度谱
    # cv2.imshow('Magnitude Spectrum', magnitude_spectrum)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 分析频域中的高频部分
    # 选择一个合适的阈值来判断高频能量是否过高
    threshold = np.mean(magnitude_spectrum) + 3 * np.std(magnitude_spectrum)
    high_frequency_pixels = magnitude_spectrum > threshold

    # 如果高频部分超过阈值，应用高斯滤波
    if np.sum(high_frequency_pixels) > 0:
        gray_filtered = cv2.GaussianBlur(gray, (7, 7), 0)
        print("高频成分较多,应用高斯滤波")
    else:
        gray_filtered = gray
        print("高频成分较少，不应用高斯滤波")

    # 计算图像的中值
    v = np.median(gray_filtered)

    # 设置Canny边缘检测的上下阈值
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    # 使用Canny边缘检测算法
    edges = cv2.Canny(gray_filtered, lower, upper)

    # 在原图上绘制边缘
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255, 0, 0), 2)

    # 将 BGR 图像转换为 RGB
    # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) (用别的库显示图像时需要进行这一步转化)

    # 使用cv2.imshow不需要转化
    rgb_image = image

    return rgb_image