# -*- coding:utf-8 -*-

import cv2
import numpy as np

#读取图片
src = cv2.imread('mask.jpg', cv2.IMREAD_UNCHANGED)

#设置卷积核
kernel1 = np.ones((4,4), np.uint8)
kernel2 = np.ones((6,6), np.uint8)
#图像膨胀处理
erosion = cv2.erode(src, kernel1)
dilation=cv2.dilate(erosion,kernel2)



#显示图像
cv2.imshow("src", src)
cv2.imshow("result", erosion)
cv2.imwrite('erosion.jpg', erosion)
cv2.imwrite('NYC_dilation.jpg', dilation)
print(dilation.shape)

#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()
