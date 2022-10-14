import numpy as np

from label import labels,number
import cv2
from skimage import measure, color
import  pandas as pd
# 取出labelsFinal作为一个数组
def excel_one_line_to_list(path, colNum):
    df = pd.read_excel(path, usecols=[colNum],
                       names=None)  # 读取项目名称列,不要列名
    df_li = df.values.tolist()
    result = []
    for s_li in df_li:
        result.append(s_li[0])
    # print(result)
    return result

# Label=labels
Label=np.zeros((685,945))
path = 'data/NEWNYC/2009NYC/4LDA_category.xlsx'
labelsFinal = excel_one_line_to_list(path, 0)

for i in range(len(labels)):
    for j in range(len((labels[i]))):
         if labels[i][j]==0:
             Label[i][j]=0
         else:
            index=labels[i][j]
            Label[i][j]=labelsFinal[index-1]+1

dst = color.label2rgb(Label, bg_label=0)    # bg_label=0要有，不然会有警告
for i in range(len(labels)):
    for j in range(len(labels[i])):
        if labels[i][j] == 1:
            dst[i][j] = 1
# print(labels[620,995])

# cv2.imshow("666", dst)
cv2.imwrite("images/NEWNYC/2009NYC/4LDA_finallabel.bmp",dst*255)  # 输出最后分类图片
cv2.waitKey(0)
cv2.destroyAllWindows()