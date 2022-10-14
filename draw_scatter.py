import pandas as pd
import matplotlib.pyplot as plt

filename = 'data/NEWNYC/2015NYC/2015TF-IDF1.xlsx'  # 读取TF-IDF矩阵
df = pd.read_excel(filename)
TFIDF = df.to_numpy()
name = ['Region-ID', 'Restaurant', 'Fastfood restaurant', 'Dessert', 'Cafe&Tea&Bar',
        'Hotel', 'Art and Science performance', 'Sports court', 'Sports&Stationery shop', 'Residence', 'Pet area',
        'Transportation facilities', 'Car&Motorcycle', 'Shopping mall', 'Apparel and Accessory store',
        'General-merchandise store', 'Electronic equipment store', 'Life service', 'Wedding&Funeral',
        'Entertainment area', 'Factory', 'Business', 'Education', 'Tourist industry', 'Open rest area',
        'Bridge&Entrance', 'Religion area', 'Hospital and Pharmacy', 'Government building', 'Office',
        'Street furniture']

nrow, ncol = TFIDF.shape
for j in range(1, 31):
    x = []
    y = []
    for i in range(nrow):
        x1 = TFIDF[:, 0]
        y1 = TFIDF[:, j]

        if y1[i] != 0:
            x.append(x1[i])
            y.append(y1[i])

    plt.figure(figsize=(10, 8), dpi=80)
    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 40,
             }
    plt.yticks(size=15, weight='bold')  # 设置坐标轴大小及加粗
    plt.xticks(size=15, weight='bold')  # 设置坐标轴大小及加粗
    plt.xlabel(name[0], font1)
    plt.ylabel(name[j], font1)
    plt.scatter(x, y, s=100)
    plt.draw()
    way = (name[j] + '.png')
    plt.savefig('./images/NEWNYC/2015NYC/scatter/%s' % way, bbox_inches='tight')
