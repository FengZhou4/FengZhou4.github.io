import pandas as pd
import akshare as ak
import plotly.graph_objects as go
import plotly.io as pio

# 下载股票数据
code='300024'
# stock_info = ak.stock_zh_a_spot()
stock_info = pd.read_csv('stock_info.csv')
stock_info['code']=stock_info['代码'].apply(lambda x: str(x)[2:].zfill(6))
name=stock_info[stock_info['code']==code]['名称'].values[0]
print(name)
pass
