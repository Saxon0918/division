import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score


class FWCM(object):
    # data 原始数据
    # c 类别数
    # p 模糊权重
    # q 模糊加权指数
    # k 最大迭代次数
    # t 聚类中心变化阈值
    def __init__(self, data, c, p, q, k, t) -> None:
        super().__init__()
        self.data = data
        self.c = c
        self.p = p
        self.q = q
        self.k = k
        self.t = t

    def classify(self):
        nrow, ncol = self.data.shape
        # 初始化隶属度
        W = np.zeros((self.c, nrow))
        for i in range(self.c):
            for j in range(nrow):
                W[i, j] = np.random.rand()
        # 隶属度之和归一化
        W = np.transpose(W.transpose() / np.sum(W, axis=1))

        # 初始化聚类中心
        C = np.zeros((self.c, ncol))
        for i in range(self.c):
            for j in range(nrow):
                C[i] += W[i, j] * self.data[j, :]
            C[i] = C[i] / np.sum(W[i:, ])

        # 初始化特征加权系数矩阵
        E = np.zeros((self.c, ncol))
        for i in range(self.c):
            entropys = list()
            for m in range(ncol):
                for j in range(nrow):
                    entropy = np.sum(W[i, j] * ((self.data[j, m] - C[i, m]) ** 2))
                entropys.append(entropy)
            entropys = np.array(entropys)
            entropys = entropys / np.sum(entropys)
            entropys = np.power(entropys, 1 / (1 - self.q))
            E[i, :] = entropys
        # 加权系数矩阵归一化
        E = np.transpose(E.transpose() / np.sum(E, axis=1))

        k = 0  # 当前迭代次数
        t = self.t * 2  # 聚类中心变化初始值
        while k < self.k and t > self.t:
            # 更新隶属度
            for i in range(nrow):
                # 计算样本到每一个聚类中心的距离
                dists = list()
                for j in range(self.c):
                    dist = np.sum(E[j, :] * (self.data[i, :] - C[j, :]) ** 2)
                    dists.append(dist)
                # 计算隶属度
                dists = np.array(dists)
                dists = dists / np.sum(dists)
                dists = np.power(dists, 1 / (1 - self.p))
                W[:, i] = dists

            # 更新聚类中心
            Ct = np.zeros((self.c, ncol))
            for i in range(self.c):
                for j in range(nrow):
                    Ct[i] += self.data[j, :] * W[i, j]
                Ct[i] = Ct[i] / np.sum(W[i, :])

                # 初始化特征加权系数矩阵
            E = np.zeros((self.c, ncol))
            for i in range(self.c):
                entropys = list()
                for m in range(ncol):
                    for j in range(nrow):
                        entropy = np.sum(W[i, j] * ((self.data[j, m] - C[i, m]) ** 2))
                    entropys.append(entropy)
                entropys = np.array(entropys)
                entropys = entropys / np.sum(entropys)
                entropys = np.power(entropys, 1 / (1 - self.q))
                E[i, :] = entropys
                # 加权系数矩阵归一化
            E = np.transpose(E.transpose() / np.sum(E, axis=1))

            t = np.max(np.abs(Ct - C))
            k += 1
            C = Ct

        # 硬化，确定最终的聚类结果
        result = np.zeros(nrow)
        for i in range(nrow):
            w = W[:, i]
            wMax = np.max(w)
            pos = np.where(w == wMax)
            result[i] = pos[0][0]
        self.result = result
        # return k
        return E


# 加载鸢尾花数据集 算法测试
# data = datasets.load_iris().data

filename = 'data/NEWNYC/2009NYC/4LDA_NOZERO.xlsx'  # 读取删除全零行列的powerlaw_LDA矩阵，进行聚类分析
df = pd.read_excel(filename, header=None)
data = df.to_numpy()

fcm = FWCM(data, 10, 2, 3.5, 10000, 0.001)  # 调用FWCM方法

# 初始化评价聚类方法
# sil = []  # 剪影值
# chs = []  # CH值
# kl = []
# beginK = 6
# endK = 16

# 计算剪影值
# for k in range(beginK, endK):
#     fcm = FWCM(data, k, 2, 3.5, 1000, 0.001)
#     fcm.classify()
#     rlabels = fcm.result
#     SC = silhouette_score(data, rlabels, metric='euclidean')
#     sil.append(SC)
#     kl.append(k)
# plt.plot(kl, sil)
# plt.ylabel('Silhoutte Score')
# plt.xlabel('K')
# plt.show()
# bestK = kl[sil.index(max(sil))]
# print("the best k is :" + str(bestK))

# # 计算CH值
# for k in range(beginK, endK):
#     fcm = FWCM(data, k, 2, 3.5, 10000, 0.001)  # 调用FWCM方法
#     fcm.classify()
#     result = fcm.result
#     result = result.astype(int)
#     score = calinski_harabasz_score(data, result)
#     print('聚类%d簇的calinski_harabaz分数为：%f' % (k, score))
#     chs.append(score)
#     kl.append(k)
# plt.plot(kl, chs)
# plt.ylabel('Calinski-Harabaz Score')
# plt.xlabel('K')
# plt.show()

# 类别权值矩阵
category_weight = fcm.classify()
category_weights = pd.DataFrame(category_weight)
category_weights.to_excel("data/NEWNYC/2009NYC/4LDA_entropy.xlsx")
# 地区类别矩阵
result = fcm.result
kind = pd.DataFrame(result)
kind.to_excel("data/NEWNYC/2009NYC/4LDA_category.xlsx")
