import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

filename = 'data/SH_powerlaw_dmr1000.xlsx'
df = pd.read_excel(filename, header=None)
arr = df.to_numpy()

p = []
k = 0
for i in range(len(arr)):
    for j in range(len(arr)):
        if (arr[i][j] != 0):
            temp = [i, j, arr[i][j]]
            p.append(temp)
            # p[k][0] = i
            # p[k][1] = j
            # p[k][2] = arr[i][j]
            # k += 1
p = np.array(p)

ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
ax.set_title('region_category')  # 设置本图名称
ax.scatter(p[:, 0], p[:, 1], p[:, 2], c='r')  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色

ax.set_xlabel('regions')  # 设置x坐标轴
ax.set_ylabel('categories')  # 设置y坐标轴
ax.set_zlabel('probability')  # 设置z坐标轴

plt.show()

