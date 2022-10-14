import csv
import json
import pandas as pd

import codecs
with open('./data/POIs.json', encoding='utf-8') as f:
    json_data=json.load(f,strict=False)
print(type(json_data))
print(json_data[1]['name'])

f = open('./data/json2csvNew.csv', 'w', encoding=' utf-8-sig',newline='')
# f.write(codecs.BOM_UTF8) # 解决中文乱码问题
csv_writer=csv.writer(f);
csv_writer.writerow(["typecode","type","name","longitude","latitude"])
for i in range(298570):
    if json_data[i]['cityname']=="上海市":
        csv_writer.writerow([json_data[i]['typecode'],json_data[i]['type'],json_data[i]['name'],json_data[i]['location'].split(',')[0],json_data[i]['location'].split(',')[1]])
f.close()
print("ok")

