# 导入第三方包
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import xlsxwriter
# 随机生成三组二元正态分布随机数 
np.random.seed(1234)
mean1 = [0.5, 0.5]
cov1 = [[0.3, 0], [0, 0.3]]
x1, y1 = np.random.multivariate_normal(mean1, cov1, 1000).T  #生成正态分布的随机数
mean2 = [0, 8]
cov2 = [[1.5, 0], [0, 1]]
x2, y2 = np.random.multivariate_normal(mean2, cov2, 1000).T
mean3 = [8, 4]
cov3 = [[1.5, 0], [0, 1]]
x3, y3 = np.random.multivariate_normal(mean3, cov3, 1000).T
# 绘制三组数据的散点图
# print(x1)


filename='data/SH_powerlaw_dmr10000.xlsx'
df = pd.read_excel(filename,  header=None)
X=df
# 将三组数据集汇总到数据框中
# X = pd.DataFrame(np.concatenate([np.array([x1,y1]),np.array([x2,y2]),np.array([x3,y3])], axis = 1).T)
# # 自定义函数的调用
# kmeans = KMeans(n_clusters=15, init='k-means++')  # 选择的是kmeans++的方法
# kmeans.fit(X)
# # 返回簇标签
# labels = kmeans.labels_
# labels=labels[np.newaxis,:]
# labels=labels.T
# # Y=np.concatenate([X,labels], axis = 1)
# print(int(labels[11]))


# 寻找最合适的K值
sil = []
kl = []
beginK = 2
endK = 20
for k in range(beginK, endK):
    kMeans = KMeans(n_clusters=k)
    kMeans.fit(X)  # 聚类
    SC = silhouette_score(X, kMeans.labels_, metric='euclidean')
    sil.append(SC)
    kl.append(k)
plt.plot(kl, sil)
plt.ylabel('Silhoutte Score')
plt.xlabel('K')
plt.show()

# k-means聚类
bestK = kl[sil.index(max(sil))]
print("the best k is :"+str(bestK))
km = KMeans(n_clusters=bestK)
km.fit(X)
# y_predict = km.predict(X)
# # 评估聚类效果
# print(silhouette_score(X, y_predict))
# 返回簇标签
labels = km.labels_
labels=labels[np.newaxis,:]
labels=labels.T
# Y=np.concatenate([X,labels], axis = 1)
print(int(labels[11]))


# labels数组存放了区域号和对应的聚类标签。最后，将这组labels写入excel即可

# 将labels（result）写入labels.xlsx中
regionsIdPath='./data/labelsfinal.xlsx'
workbook = xlsxwriter.Workbook(regionsIdPath)
worksheet = workbook.add_worksheet()
row1 = 0
col1 = 0
for  data in labels:
    # print(data)
    worksheet.write(row1, col1, data)
    row1=row1+1

workbook.close()
