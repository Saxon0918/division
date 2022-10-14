import numpy as np
import pandas as pd

filename1= 'data/NEWNYC/2015NYC/4NYC_1000NOZERO_cate.xlsx'  # 读取删除全零行列的powerlaw_LDA矩阵，进行区域权值计算
df1 = pd.read_excel(filename1, header=None)
nozero=df1.to_numpy()

filename2= 'data/NEWNYC/2015NYC/4NYC_1000entropy.xlsx'  # 读取种类权值矩阵，进行区域权值计算
df2 = pd.read_excel(filename2, header=None)
entropy=df2.to_numpy()

nrow, ncol = nozero.shape
category = list()
topic = np.zeros((nrow,ncol-1))
for i in range(ncol):
    if i==0:
        category = nozero[:,i]
    else:
        topic[:,i-1] = nozero[:,i]

weight = []
for i in range(nrow):
    weight.append(np.dot(topic[i,:],entropy[int(category[i]),:]))  # 计算每一个区域的权值

weights = pd.DataFrame(weight)
weights.to_excel("data/NEWNYC/2015NYC/4NYC_1000regionweight.xlsx")
# print(weight)