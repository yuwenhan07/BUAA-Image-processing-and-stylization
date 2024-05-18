import cv2
import matplotlib.pyplot as plt
import numpy as np

def sharpen_faces(image):
    # 定义一个函数来对图像进行锐化
    def sharpen_image(image):
        # 定义拉普拉斯滤波器
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        # 使用滤波器对图像进行卷积
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    # 加载预训练的Haar级联分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 深拷贝原始图像
    img_copy = image.copy()

    # 转换为灰度图，提高检测效率
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 识别图像中的人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 在原图上绘制人脸区域并对人脸图像进行锐化
    for (x, y, w, h) in faces:
        # 截取人脸部分
        face_img = image[y:y+h, x:x+w]
        # 锐化人脸图像
        sharpened_face = sharpen_image(face_img)
        # 将锐化后的人脸图像放回原图像的相应位置
        img_copy[y:y+h, x:x+w] = sharpened_face

    return img_copy