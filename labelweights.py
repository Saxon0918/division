# 标记区域
import cv2
from skimage import measure, color
import pandas as pd
import numpy as np
path = "images/NEWNYC/NYC_thinned.bmp"
img = cv2.imread(path)
img_copy = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gauss = cv2.GaussianBlur(img_gray, (5, 5), 1)
img_temp = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)[1]
labels,number = measure.label(img_temp,connectivity=1,return_num =True)
print("number:"+str(number))
# print("labels:"+labels)
dst = color.label2rgb(labels, bg_label=0)    # bg_label=0要有，不然会有警告
# print(labels[620,995])
a = measure.regionprops(labels)
font = cv2.FONT_HERSHEY_SIMPLEX
def excel_one_line_to_list(path, colNum):
    df = pd.read_excel(path, usecols=[colNum],
                       names=None)  # 读取项目名称列,不要列名
    df_li = df.values.tolist()
    result = []
    for s_li in df_li:
        result.append(s_li[0])
    # print(result)
    return result
path = 'data/NEWNYC/2015NYC/4NYC_1000category.xlsx'
type = excel_one_line_to_list(path, 0)  # 读取区域类别
weight= excel_one_line_to_list(path, 2)  # 区域区域权值
# type2weight=[[0 for i in range(2)] for i in range(len(type))]
# for i in range(len(type)):
#         type2weight[i][0]=type[i]
#         type2weight[i][1]=weight[i]
Label=np.zeros((685,945))
Weights=np.zeros((685,945))
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

for (j, i) in enumerate(a):
    x,y = i.centroid
    l=i.label
    x=int(x)
    y=int(y)
    L = Label[x, y]
    w=Weights[x, y]
    # cv2.putText(img_copy, str(l), (y, x), font, 0.2, (128, 0, 128), 1)  # 区域号
    # cv2.putText(img_copy, str(L), (y, x), font, 0.2, (0, 0, 255), 1)  # 类别号
    cv2.putText(img_copy, str(w), (y, x ), font, 0.2, (255, 0, 0), 1)  # 区域权值

img = cv2.resize(img_copy,None,fx=2,fy=2)  # 设置图片尺寸*2后再显示
cv2.imshow("image", img)
cv2.imwrite("images/NEWNYC/2015NYC/img_copy.bmp",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
