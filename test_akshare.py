import pandas as pd
import akshare as ak
import plotly.graph_objects as go
import plotly.io as pio

# # 下载股票数据
# code='300024'
# # stock_info = ak.stock_zh_a_spot()
# stock_info = pd.read_csv('stock_info.csv')
# stock_info['code']=stock_info['代码'].apply(lambda x: str(x)[2:].zfill(6))
# name=stock_info[stock_info['code']==code]['名称'].values[0]
# print(name)
# pass

# csi_300_df = ak.index_stock_cons("000300")
# csi_300_df.to_hdf('csi_300_df.h5', key='df', mode='w')
# for id, stock_code in csi_300_df["品种代码"].items():
#     pass


# csi_300_df= pd.read_hdf('csi_300_df.h5', key='df')
# a=csi_300_df["品种代码"].values.tolist()
# pass


# csi_300_df= pd.read_csv('csi_300_df.csv')
# a=csi_300_df["品种代码"].values.tolist()
# pass

news = ak.stock_news_em(symbol='600519').head(5)
news.to_csv('news.csv', index=False)
print(news)











