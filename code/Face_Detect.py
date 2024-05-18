import cv2
import matplotlib.pyplot as plt

def detect_and_extract_faces(image):
    # 加载预训练的Haar级联分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 转换为灰度图，提高检测效率
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 识别图像中的人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 创建一个列表以存储每个人脸的图像
    face_images = []

    # 在原图上绘制人脸区域
    for (x, y, w, h) in faces:
        # 截取人脸部分
        face_img = image[y:y+h, x:x+w]
        # 由于OpenCV使用BGR格式，而matplotlib使用RGB格式，需要转换颜色通道
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # 添加到列表中
        face_images.append(face_img_rgb)
        # 在原图上绘制矩形
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # 由于OpenCV使用BGR格式，而matplotlib使用RGB格式，需要转换颜色通道
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image_rgb, face_images
