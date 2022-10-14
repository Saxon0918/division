from label import labels
import pandas as pd
import xlsxwriter
from PIL import Image


def excel_one_line_to_list(path, colNum):
    df = pd.read_excel(path, usecols=[colNum],
                       names=None)  # 读取项目名称列,不要列名
    df_li = df.values.tolist()
    result = []
    for s_li in df_li:
        result.append(s_li[0])
    # print(result)
    return result


path = 'data/NYC/2009NYC/2011NYCPOI.xlsx'
row = excel_one_line_to_list(path, 4)
col = excel_one_line_to_list(path, 5)
result = []
for i in range(len(row)):
    #     将labels[row[i]][col[i]]输出
    #     print(labels[row[i]][col[i]])
    result.append(labels[row[i]][col[i]])

# 将regionsId（result）写入regionsId.xlsx中
regionsIdPath='data/regionsId.xlsx'
workbook = xlsxwriter.Workbook(regionsIdPath)
worksheet = workbook.add_worksheet()
row1 = 0
col1 = 0
for  data in result:
    # print(data)
    worksheet.write(row1, col1, data)
    row1=row1+1

workbook.close()

# img = Image.open("NYC_thinned.bmp")#读取系统的内照片
# for i in range(len(row)):
#     #     将labels[row[i]][col[i]]输出
#     #     print(labels[row[i]][col[i]])
#     img.putpixel((row[i], col[i]), (234, 53, 57))  # 则这些像素点的颜色改成大红色
# img = img.convert("RGB")#把图片强制转成RGB
# img.save("testee1.jpg")#保存修改像素点后的图片