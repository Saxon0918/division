import numpy as np
from scipy.stats import norm
import pandas as pd
import math
from TFIDFCF import U_new

# X = np.arange(1, 376)  # 1-376，每隔着1一个数据  0处取对数,会时负无穷  生成375个数据点
# print(X[0])
# noise=norm.rvs(0, size=400, scale=0.2)  # 生成50个正态分布  scale控制噪声强度
# Powerlow_k=[]
# for i in range(len(X)):
#     Powerlow_k.append(((10.8*pow(X[i],-0.3)+noise[i])/10))  # 得到Y=10*x^-0.3+noise
# print(Powerlow_k) # plot raw data
# # Y=np.array(Powerlow_k)
#
# Powlaw_num = pd.DataFrame(Powerlow_k)
# Powlaw_num.to_excel("data/NEWNYC/Powlaw_num.xlsx")  # 将随机Powerlaw写入文件

path = 'data/NEWNYC/Powlaw_num.xlsx'
def excel_one_line_to_list(path, colNum):
    df = pd.read_excel(path, usecols=[colNum],
                       names=None)  # 读取项目名称列,不要列名
    df_li = df.values.tolist()
    result = []
    for s_li in df_li:
        result.append(s_li[0])
    # print(result)
    return result
Powerlow_k = excel_one_line_to_list(path, 0)
print(Powerlow_k)

# filename='data/NEWNYC/U_new.xlsx'  # 读取出发-到达矩阵 删除属性列和行
# df = pd.read_excel(filename, header=None)
# arr=df.to_numpy().T

arr=U_new
arr = np.array(arr)
arr = arr.T
alpha_i_k = []
for i in range(len(arr)):
    alpha_i_k.append(math.exp(np.dot(arr[i], Powerlow_k)))   # 计算每个区域的alpha值
    # print(alpha_i_k)

Arrive_leave = pd.DataFrame(alpha_i_k)
Arrive_leave.to_excel("data/NEWNYC/2015NYC/2015NYC_alpha_i_k.xlsx")  # 将每个区域的alpha值写入文件

# plt.title("Raw data")
# plt.scatter(X, Y,  color='black')
# plt.show()