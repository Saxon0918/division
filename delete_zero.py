import numpy as np
import pandas as pd

filename= 'data/NEWNYC/2009NYC/4LDA.xlsx'  # 读取powerlaw_LDA矩阵，删除全0行列
df = pd.read_excel(filename, header=None)
data=df.to_numpy()

# 删除全0行
newdata_row = []
rownum = list()  # 记录全0行号
nrow, ncol = data.shape  # 记录行列号
for i in range(nrow):
    rowsum = 0
    for j in range(ncol):
        rowsum += data[i][j]
    if rowsum != 0:
        newdata_row.append(data[i,:])  # 将非全0行数据存入newdata矩阵
    if rowsum == 0:
        rownum.append(i)
print(rownum)
newdata_row=np.array(newdata_row)
newrow, newcol = newdata_row.shape

# 删除全0列
newdata_col = []
for i in range(ncol):
    colsum = 0
    for j in range(newrow):
        colsum += newdata_row[j][i]
    if colsum != 0:
        newdata_col.append(newdata_row[:, i])
newdata_col=np.array(newdata_col)
newdata_col = newdata_col.T
# print(newdata_col)

# 数据存入excel
newdata = pd.DataFrame(newdata_col)
newdata.to_excel("data/NEWNYC/2009NYC/4LDA_NOZERO.xlsx")