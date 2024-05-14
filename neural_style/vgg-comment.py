from collections import namedtuple

import torch
from torchvision import models

class Vgg16(torch.nn.Module):  # 定义Vgg16类，继承自torch.nn.Module
    def __init__(self, requires_grad=False):  # 初始化函数，requires_grad决定是否冻结参数
        super(Vgg16, self).__init__()  # 调用父类的初始化函数
        vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features  # 加载预训练的VGG16模型特征部分
        self.slice1 = torch.nn.Sequential()  # 定义第一个子模块slice1
        self.slice2 = torch.nn.Sequential()  # 定义第二个子模块slice2
        self.slice3 = torch.nn.Sequential()  # 定义第三个子模块slice3
        self.slice4 = torch.nn.Sequential()  # 定义第四个子模块slice4
        for x in range(4):  # 将前4层添加到slice1
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):  # 将第5到第9层添加到slice2
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):  # 将第10到第16层添加到slice3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):  # 将第17到第23层添加到slice4
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:  # 如果requires_grad为False，冻结所有参数
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):  # 定义前向传播函数
        h = self.slice1(X)  # 输入X通过slice1
        h_relu1_2 = h  # 获取slice1的输出
        h = self.slice2(h)  # 输出通过slice2
        h_relu2_2 = h  # 获取slice2的输出
        h = self.slice3(h)  # 输出通过slice3
        h_relu3_3 = h  # 获取slice3的输出
        h = self.slice4(h)  # 输出通过slice4
        h_relu4_3 = h  # 获取slice4的输出
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])  # 定义输出命名元组
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)  # 组织输出
        return out  # 返回提取的特征
