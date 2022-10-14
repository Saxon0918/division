# -*- coding:utf-8 -*-

import cv2 as cv
import numpy as np


# 全局阈值
def threshold_demo(image):
  gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY) #把输入图像灰度化
  #直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
  ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
  print("全局阈值：threshold value %s"%ret)
  cv.namedWindow("binary0", cv.WINDOW_NORMAL)
  cv.imshow("binary0", binary)
  # print(binary.shape)
  # a = binary.shape[0]
  # b = binary.shape[1]
  # print(a, b)
  # for i in range(0, a):
  #   for j in range(0, b):
  #     print(binary[i][j])

def custom_Threshold(img):

  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  h, w = gray.shape[:2]
  m = np.reshape(gray, [1, h * w])  # 将图像转为1行h*w列
  mean = m.sum() / (h * w)  # 计算图像的均值，用均值作为阈值，来分割图像
  ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
  print("自定义阈值Threshold value %s" % ret)
  cv.imshow("custom_binary", binary)
  cv.imwrite('mask.jpg', binary)
  cv.waitKey(0)
  cv.destroyAllWindows()
  print(binary.shape)
  # a = binary.shape[0]
  # b = binary.shape[1]
  # print(a, b)
  # for i in range(0, a):
  #   for j in range(0, b):
  #     print(binary[i][j])


#局部阈值
def local_threshold(image):
  gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY) #把输入图像灰度化
  #自适应阈值化能够根据图像不同区域亮度分布，改变阈值
  binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 25, 10)
  cv.namedWindow("binary1", cv.WINDOW_NORMAL)
  cv.imshow("binary1", binary)
  # cv.imwrite('mask.jpg', binary)
  # a = binary.shape[0]
  # b = binary.shape[1]
  # print(a, b)
  # for i in range(0, a):
  #   for j in range(0, b):
  #     print(binary[i][j])



#用户自己计算阈值
# def custom_threshold(image):
#   gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY) #把输入图像灰度化
#   h, w =gray.shape[:2]
#   m = np.reshape(gray, [1,w*h])
#   mean = m.sum()/(w*h)
#   print("mean:",mean)
#   ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
#   cv.namedWindow("binary2", cv.WINDOW_NORMAL)
#   cv.imshow("binary2", binary)

src = cv.imread('NYC-Map.png')
# print(type(src))
cv.namedWindow('input_image', cv.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
cv.imshow('input_image', src)
threshold_demo(src)
local_threshold(src)
custom_Threshold(src)

cv.waitKey(0)
cv.destroyAllWindows()