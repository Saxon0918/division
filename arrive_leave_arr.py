import numpy as np
import pandas as pd

number=375
time_zone=12
path = 'data/NYC/2015NYC/2015NYCTAXI.xlsx'

def excel_one_line_to_list(path, colNum):
    df = pd.read_excel(path, usecols=[colNum],
                       names=None)  # 读取项目名称列,不要列名
    df_li = df.values.tolist()
    result = []
    for s_li in df_li:
        result.append(s_li[0])
    # print(result)
    return result

startTimeZone = excel_one_line_to_list(path, 6)
arriveTimeZone = excel_one_line_to_list(path, 7)
startRegion = excel_one_line_to_list(path, 8)
arriveRegion = excel_one_line_to_list(path, 9)

Arrive=[[0 for i in range(4)] for i in range(len(startRegion))]
for i in range(len(startRegion)):
        Arrive[i][0]=startRegion[i]
        Arrive[i][1]=arriveRegion[i]
        Arrive[i][2] = startTimeZone[i]
        Arrive[i][3] =arriveTimeZone[i]
# print(Arrive)

Arrive_arr=np.zeros((number,number*time_zone))
Leave_arr=np.zeros((number,number*time_zone))

for k in range(len(Arrive)):
    Arrive_arr[Arrive[k][1]][time_zone*Arrive[k][0]+Arrive[k][3]] += +1
    Leave_arr[Arrive[k][0]][time_zone*Arrive[k][1]+Arrive[k][2]] += +1

Arrive_leave = np.hstack((Arrive_arr,Leave_arr))

Arrive_leave=pd.DataFrame(Arrive_leave.T)
Arrive_leave.to_excel("data/NYC/2015NYC/2015NYC_4trajectory.xlsx")
