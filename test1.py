import talib
print(talib.__version__)

import numpy as np

# 示例数据：收盘价（假设为连续10天的数据）
close_prices = np.array([45.3,46.1,47.8,48.5,49.2,50.1,49.8,51.0,50.5,52.3])

# 计算5日简单移动平均线
sma5 = talib.SMA(close_prices, timeperiod=5)
print("5日SMA:", sma5)
# 输出：[       nan        nan        nan        nan 47.38, 48.34, 49.12, 49.88, 50.32, 50.94]



rsi14 = talib.RSI(close_prices, timeperiod=14)
print("14日RSI:", rsi14[-1])  # 输出最新RSI值

# 策略示例：RSI低于30时触发买入信号
if rsi14[-1] < 30:
    print("买入信号：RSI超卖区域！")


#MACD由快线（DIF）、慢线（DEA）和柱状图（MACD Histogram）组成，常用于判断多空转换。
macd, signal, hist = talib.MACD(close_prices, 
                               fastperiod=12, 
                               slowperiod=26, 
                               signalperiod=9)
# 金叉：MACD线上穿信号线（买入信号）
if macd[-2] < signal[-2] and macd[-1] > signal[-1]:
    print("MACD金叉出现，建议买入！")


#布林带由中轨（均线）、上轨和下轨组成，价格突破轨道常被视为趋势信号。
upper, middle, lower = talib.BBANDS(close_prices, 
                                    timeperiod=20, 
                                    nbdevup=2, 
                                    nbdevdn=2)
# 价格突破上轨可能预示回调
if close_prices[-1] > upper[-1]:
    print("价格突破布林带上轨，警惕超买风险！")

















