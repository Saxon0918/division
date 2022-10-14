import numpy as np
import pandas as pd
import math
from numpy import random
from TFIDFCF import U_new

sampleNo = 30  # 向量纬度

mu = 0
sigma = 1
np.random.seed(0)
Lambda_k = np.random.normal(mu, sigma, sampleNo)  # 生成Lambda_k向量
print(Lambda_k)

arr=U_new

alpha_i_k = []
for i in range(len(arr)):
    alpha_i_k.append(math.exp(np.dot(arr[i], Lambda_k.T)))   # 计算每个区域的alpha值
    print(alpha_i_k)

Arrive_leave = pd.DataFrame(alpha_i_k)
Arrive_leave.to_excel("data/SH_alpha_i_k.xlsx")  # 将每个区域的alpha值写入文件