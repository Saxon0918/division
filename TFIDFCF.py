import math

from label import number
import pandas as pd
import numpy as np

filename = 'data/NEWNYC/2009NYC/2011NYCPOI.xlsx'

# 通过read_excel()方法生成DataFrame对象

df = pd.read_excel(filename, header=None)
# print(df.iloc[1:][5])
typeId = []
regionsId = []
for i in range(1, df.shape[0]):
    typeId.append(df.iloc[i][3])
    regionsId.append(df.iloc[i][6])
# print(typeId)
# print(regionsId)

# print("##################")
# print(len(typeId))
# 将typeId和regionsId合并成一个新的矩阵,这是poi的种类id和对应的区域id
TypeRegions = [[0 for i in range(2)] for i in range(len(typeId))]
for i in range(len(typeId)):
    TypeRegions[i][0] = typeId[i]
    TypeRegions[i][1] = regionsId[i]
# print(TypeRegions)
# 对于每一个区域，要计算它拥有的第j类poi的数目（二维）,注意j从1开始。维度：number*100
# 第i个区域中poi的总数
Total = [0 for i in range(number-1)]
region2poi = np.zeros((number-1, 30))
for i in range(len(typeId)):
    if (TypeRegions[i][1]-1):
        region2poi[TypeRegions[i][1] - 2][TypeRegions[i][0] - 1] += 1
        Total[TypeRegions[i][1] - 2] += 1

# 对于每一类poi，要计算拥有它的区域总数（一维）
poi2region = []
for i in range(30):
    # 存放拥有该类poi的区域的区域号
    list = []
    for j in range(len(TypeRegions)):
        if (TypeRegions[j][0] == (i + 1)):
            list.append(TypeRegions[j][1])

    poi2region.append(len(set(list)))
print(poi2region)

# 计算TF-IDF
tfidf = np.zeros((number-1,30))
    # [[0 for i in range(30)] for i in range(number-1)]
U_new = []
for i in range(number-1):
    for j in range(30):
        tf = region2poi[i][j] / (Total[i] + 1)
        idf = math.log10(number / (poi2region[j] + 1))
        tfidf[i][j] = tf * idf
    U_new.append(tfidf[i,:])

U_new_num = pd.DataFrame(U_new)
U_new_num.to_excel("data/NEWNYC/2009NYC/2009TF-IDF.xlsx")  # 将TF-IDF矩阵写入文件
# print("tfidf:")
# print(tfidf)

# svd分解
# print("######################C-F#########################")
# F = np.array(tfidf)  # 计算SVD
# U, S, V = np.linalg.svd(F, full_matrices=False)  # 计算SVD
# print(U.shape)
# print(S.shape)
# print(V.shape)
# print(S)
# U_new就是CF矩阵（729×30）
# U_new = U[:, :30]  # 计算SVD
# print(U_new.shape)
# print("################CF矩阵：###########")
# print(U_new)
# TFIDFCF = pd.DataFrame(U_new)
# TFIDFCF.to_excel("data/SH_TFIDFCF.xlsx")
# print(TFIDFCF)
