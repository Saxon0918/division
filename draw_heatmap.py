from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

filename = 'data/NEWNYC/2015NYC/4NYC_1000entropy.xlsx'  # 读取entropy矩阵
df = pd.read_excel(filename)
cate_entropy = df.to_numpy()
# 绘制热度图：
plt.figure(dpi=200)
plot = sns.heatmap(cate_entropy,cmap="Oranges",linewidths=0.1)
plt.xlabel("Dimension",size=15)
plt.ylabel("Region Category",size=15)
# plt.title("heatmap",size=18)
# plt.show()
plt.savefig('./images/NEWNYC/2015NYC/2015entropy_heatmap.png', bbox_inches='tight')