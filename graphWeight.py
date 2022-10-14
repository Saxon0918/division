import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from skimage import measure, color
path = "images/NEWNYC/NYC_thinned.bmp"
img = cv2.imread(path)
img_copy = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gauss = cv2.GaussianBlur(img_gray, (5, 5), 1)
img_temp = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)[1]
labels,number = measure.label(img_temp,connectivity=1,return_num =True)
def excel_one_line_to_list(path, colNum):
    df = pd.read_excel(path, usecols=[colNum],
                       names=None)  # 读取项目名称列,不要列名
    df_li = df.values.tolist()
    result = []
    for s_li in df_li:
        result.append(s_li[0])
    # print(result)
    return result
path = 'data/NEWNYC/2009NYC/4NYC_100category.xlsx'
type = excel_one_line_to_list(path, 0)  # 读取区域类别
weight= excel_one_line_to_list(path, 2)  # 区域区域权值
Label=np.zeros((685,945))
Weights=np.zeros((685,945))
weights=np.zeros((685,945))

#
for i in range(len(labels)):
    for j in range(len((labels[i]))):
         if labels[i][j]==0:
             Label[i][j]=0
             Weights[i][j]=0
         else:
            index=labels[i][j]
            Label[i][j]=type[index-1]+1
            Weights[i][j] =weight[index-1]

for i in range(685):
    for j in range(945):
        weights[i][j]=Weights[684-i][j]
###### 数据X，Y，Z
x = np.arange(945) # 步长为0.01，即每隔0.01取一个点
y = np.arange(685) # 步长为0.01，即每隔0.01取一个点
X, Y = np.meshgrid(x, y)     # 将原始数据变成网格数据形式
Z = weights             # 我们假设Z为关于X，Y的函数，以Z=X^2+Y^2为例
# ctf = plt.contourf(X,Y,Z,15)
# ct = plt.contour(X,Y,Z,15,colors='k')    # 等高线设置成黑色
# plt.clabel(ct, inline=True, fontsize=10) # 添加标签
# # plt.pcolormesh(X, Y, Z)     # 绘制分类背景图
# plt.colorbar(ctf)  # 添加cbar

plt.figure(figsize=(10, 8), dpi=100)
plt.pcolormesh(X, Y, Z)     # 绘制分类背景图

# plt.xticks(())  # 去掉x标签
# plt.yticks(())  # 去掉y标签

# plt.savefig('images/NYC/NYC_2009.png')
plt.show()
