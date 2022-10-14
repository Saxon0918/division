import pandas as pd

df = pd.read_csv("data/shanghai_taxi_data.csv", error_bad_lines=False)
df.iloc[0][6] = 0
df.iloc[1][6] = 0
result = pd.DataFrame(index=range(1), columns=["startLong", "startLat", "startTime", "endLong", "endLat", "endTime"])

for i in range(len(df) - 1):
    if df.iloc[i][6] == 0 and df.iloc[i+1][6] != 0:
        taxi_num = df.iloc[i+1][0]
        startLong = df.iloc[i+1][2]
        startLat = df.iloc[i+1][3]
        startTime = df.iloc[i+1][1]
        print(startLong)

    if df.iloc[i][6] != 0 and df.iloc[i+1][6] == 0 and df.iloc[i+1][0] == taxi_num:
        endLong = df.iloc[i+1][2]
        endLat = df.iloc[i+1][3]
        endTime = df.iloc[i+1][1]
        print(endLong)
        result.loc[result.index.max()+1] = [startLong, startLat, startTime, endLong, endLat, endTime]

result.to_excel("data/shanghai_trajectory.xlsx")
