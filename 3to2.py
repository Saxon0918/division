import cv2
import numpy as np
# 三维图像转为二维图像
src=cv2.imread("NYC_dilation.jpg")
# 复制图片
img = src.copy()

img1=img[:,:,0]
print(img.shape)
print(img1.shape)
a=img.shape[0]
b=img.shape[1]
print(a,b)
for i in range(0,a) :
    for j in range(0,b):
        for k in range(1,4):
            # flag=False
            # if(img[i][j][k])==255:
            #     flag=True
            print(img[i][j])
print(flag)
# print(img1[1300][1])
# cv2.imwrite("./images/dilation2.jpg",img1)
cv2.imshow('input_image', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()