import gensim
from gensim import corpora
import numpy as np
import pandas as pd

# 375篇文档，375×24×2种词
topicnum = 30
words = 375 * 12 * 2

filename = 'data/NEWNYC/2009NYC/2009NYC_4trajectory.xlsx'  # 读取出发-到达矩阵 删除属性列和行
df = pd.read_excel(filename, header=None)
arr = df.to_numpy().T

filename_alpha = 'data/NEWNYC/2009NYC/2009NYC_alpha_i_k.xlsx'  # 读取alpha矩阵 不用删除属性列和行
df_alpha = pd.read_excel(filename_alpha)
alpha_i_k = df_alpha['alpha/100'].tolist()
print(alpha_i_k)

matrix = [i for i in range(words)]
str1 = np.array(matrix)
word_list = str1.astype('str')  # 生成word
bow_corpus = []

# dictionary = gensim.corpora.Dictionary([word_list]) #封装方法创建dictionary
# print(dictionary)
text_tokens = [[text for text in word_list.split()] for word_list in word_list]  # 按照顺序创建dictionary
dictionary = corpora.Dictionary(text_tokens)

for i in range(len(arr)):
    bow_corpus.append(list(zip(dictionary, arr[i])))  # 将dictionary与出现次数矩阵相关联

if __name__ == '__main__':
    PDR = gensim.models.LdaMulticore(bow_corpus, num_topics=topicnum, id2word=dictionary, alpha=alpha_i_k, passes=1,
                                     workers=2)
    # 执行DMR 其中 alpha_i_k 根据高斯函数根据各个位置计算
    # lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=topicnum, id2word=dictionary, passes=1, workers=2)#执行lda
    # for idx, topic in lda_model.print_topics(-1):
    #     print('Topic: {} \nWords: {}'.format(idx, topic))  # 计算每种主题-单词分布
    corpus_lda = PDR[bow_corpus]  # 计算每篇文章-主题分布
    dmr_num = np.zeros((len(arr), topicnum))  # 讲文章-主题分布写入excel
    for i in range(len(arr)):
        list = corpus_lda[i]
        for j in range(len(list)):
            index = list[j][0]
            value = list[j][1]
            dmr_num[i][index] = value

    dmr_num = pd.DataFrame(dmr_num)
    dmr_num.to_excel("data/NEWNYC/2009NYC/4LDA.xlsx")
