import os
import csv
import numpy as np

# path = "C:/Users/94376/PycharmProjects/regions/data/NYC2008_taxi/txt"  # 文件夹目录
files = "D:/PycharmProjects/regions/data/NYCPOI/New York.txt"  # 文件目录
# files = os.listdir(path)  # 得到文件夹下的所有文件名称
txts = []

csvFile = open("D:/PycharmProjects/regions/data/NYCPOI/New York.csv", 'w', newline='', encoding='gbk')  # 固定格式
writer = csv.writer(csvFile)  # 固定格式
csvRow = []  # 用来存储csv文件中一行的数据
# 对csvRow通过append()或其它命令添加数据
writer.writerow(csvRow)  # 将csvRow中数据写入csv文件中
# csvFile.close()

# for file in files:
#     # position = path + '\\' + file
#     position = file
#     print(position)
#     # with open(position, "r", encoding='utf-8') as f:  # 打开文件
#     with open(position, "r") as f:  # 打开文件
#         for line in f:
#             csvRow = line.strip().split('|')  # strip函数删除最后回车 split函数按照‘，’划分数据
#             writer.writerow(csvRow)

position = files
print(position)
# with open(position, "r", encoding='utf-8') as f:  # 打开文件
with open(position, "r") as f:  # 打开文件
    for line in f:
        csvRow = line.strip().split()  # strip函数删除最后回车 split函数按照‘，’划分数据
        writer.writerow(csvRow)
