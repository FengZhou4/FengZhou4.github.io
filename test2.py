



# 数据获取
def get_stock_data(_symbol, start, end, period_type):
    try:
        df = ak.stock_zh_a_hist(
            symbol=_symbol,
            period=period_type,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq"
        )
        if df.empty:
            return pd.DataFrame()
        df['日期'] = pd.to_datetime(df['日期'])
        return df.sort_values('日期').set_index('日期')
    except Exception as e:
        st.error(f"数据获取失败: {str(e)}")
        return pd.DataFrame()


# 技术指标计算
def calculate_indicators(df):
    # 均线系统
    df['MA5'] = talib.SMA(df['收盘'], timeperiod=5)
    df['MA10'] = talib.SMA(df['收盘'], timeperiod=10)
    df['MA20'] = talib.SMA(df['收盘'], timeperiod=20)

    # MACD
    df['MACD'], df['MACDsignal'], df['MACDhist'] = talib.MACD(
        df['收盘'], fastperiod=12, slowperiod=26, signalperiod=9)

    # RSI
    df['RSI14'] = talib.RSI(df['收盘'], timeperiod=14)

    # KDJ
    df['slowk'], df['slowd'] = talib.STOCH(
        df['最高'], df['最低'], df['收盘'],
        fastk_period=9, slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0
    )
    df['slowj'] = 3 * df['slowk'] - 2 * df['slowd']

    # 布林带
    df['upper'], df['middle'], df['lower'] = talib.BBANDS(df['收盘'], timeperiod=20)

    # 成交量
    df['VOL_MA5'] = talib.SMA(df['成交量'], timeperiod=5)
    return df.dropna()


# 创建交互图表
def create_plotly_chart(df, period):
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.15, 0.2],
        specs=[[{"secondary_y": True}], [{}], [{}], [{}], [{}]]
    )

    # K线图（红涨绿跌）
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['开盘'],
        high=df['最高'],
        low=df['最低'],
        close=df['收盘'],
        name='K线',
        increasing={'line': {'color': 'red'}, 'fillcolor': 'rgba(255,0,0,0.3)'},
        decreasing={'line': {'color': 'green'}, 'fillcolor': 'rgba(0,128,0,0.3)'}
    ), row=1, col=1)

    # 均线系统
    for ma, color in zip(['MA5', 'MA10', 'MA20'], ['orange', 'blue', 'purple']):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[ma],
            name=ma,
            line=dict(color=color, width=1.5),
            opacity=0.8
        ), row=1, col=1)

    # MACD
    colors = np.where(df['MACDhist'] > 0, 'red', 'green')
    fig.add_trace(go.Bar(
        x=df.index, y=df['MACDhist'],
        name='MACD Hist',
        marker_color=colors
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD'],
        line=dict(color='blue'),
        name='MACD'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MACDsignal'],
        line=dict(color='orange'),
        name='Signal'
    ), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI14'],
        line=dict(color='purple'),
        name='RSI 14'
    ), row=3, col=1)
    fig.add_hline(y=30, line=dict(color='gray', dash='dash'), row=3, col=1)
    fig.add_hline(y=70, line=dict(color='gray', dash='dash'), row=3, col=1)

    # KDJ
    fig.add_trace(go.Scatter(
        x=df.index, y=df['slowk'],
        line=dict(color='blue'),
        name='K值'
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['slowd'],
        line=dict(color='orange'),
        name='D值'
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['slowj'],
        line=dict(color='green', dash='dot'),
        name='J值'
    ), row=4, col=1)
    fig.add_hline(y=20, line=dict(color='gray', dash='dash'), row=4, col=1)
    fig.add_hline(y=80, line=dict(color='gray', dash='dash'), row=4, col=1)

    # 成交量
    colors = np.where(df['收盘'] > df['开盘'], 'red', 'green')
    fig.add_trace(go.Bar(
        x=df.index, y=df['成交量'],
        name='成交量',
        marker_color=colors,
        opacity=0.7
    ), row=5, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['VOL_MA5'],
        line=dict(color='orange'),
        name='成交量MA5'
    ), row=5, col=1)

    # 布局设置
    fig.update_layout(
        height=1200,
        title=f'{symbol} {period}级别技术分析',
        font=CHINESE_FONT,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode="x unified"
    )

    # Y轴标签
    yaxis_labels = ["价格", "MACD", "RSI", "KDJ", "成交量"]
    for i, label in enumerate(yaxis_labels, 1):
        fig.update_yaxes(title_text=label, row=i, col=1)

    return fig


