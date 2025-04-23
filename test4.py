import pandas as pd
import akshare as ak
import plotly.graph_objects as go
import plotly.io as pio

# 下载股票数据
 
# 获取贵州茅台的日线行情数据
data = ak.stock_zh_a_daily(symbol="sh600519",start_date='20230101',end_date='20231231', adjust="hfq")
print(data)
df = data[['open', 'high', 'low', 'close']].copy()

# 参数设置
window = 20  # 移动平均窗口
k = 2        # 标准差倍数

# 计算中轨（20日SMA）
df['MA20'] = df['close'].rolling(window).mean()

# 计算标准差
df['std'] = df['close'].rolling(window).std()

# 计算上轨和下轨
df['Upper'] = df['MA20'] + k * df['std']
df['Lower'] = df['MA20'] - k * df['std']

# 删除NaN值（前20天无数据）
df.dropna(inplace=True)

# 创建Figure对象
fig = go.Figure()

# 添加K线图
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    name='K线'
))

# 添加布林带中轨
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['MA20'],
    line=dict(color='blue', width=1.5),
    name='中轨 (MA20)'
))

# 添加上轨和下轨，并填充区域
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['Upper'],
    line=dict(color='gray', width=1),
    name='上轨'
))
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['Lower'],
    line=dict(color='gray', width=1),
    fill='tonexty',  # 填充到下一条轨迹（上轨）
    fillcolor='rgba(128,128,128,0.2)',
    name='下轨'
))

# 调整布局
fig.update_layout(
    title='贵州茅台股价与布林带（2023年）',
    xaxis_title='日期',
    yaxis_title='价格',
    hovermode='x unified',
    showlegend=True,
    template='plotly_white'  # 使用白色主题
)

# 隐藏非交易日的空白（如周末）
fig.update_xaxes(rangebreaks=[dict(bounds=['sat', 'mon'])], rangeslider_visible=True)

# # 显示图表
# fig.show()

# 导出为HTML文件（完整独立文件）
pio.write_html(fig, file='bollinger_bands.html', auto_open=True)

# # 直接调用Figure的write_html方法
# fig.write_html('bollinger_bands.html', include_plotlyjs=True)





