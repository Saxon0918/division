import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import pyodbc

def read_pyarrow(path, nthreads=1):
    return pq.read_table(path).to_pandas()

path = 'data/NYC_taxi/yellow_tripdata_2009-03.parquet'
df1 = read_pyarrow(path)

df1.to_csv(
    'data/NYC_taxi/yellow_tripdata_2009-03.csv',
    sep='|',
    index=False,
    mode='w',
    line_terminator='\n',
    encoding='utf-8')