import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

today = datetime.now()
default_start = today - timedelta(days=365)
default_end = today

# Streamlit界面设置
st.set_page_config(page_title="股票预测", layout="wide")
st.title('股票涨跌预测xgboost')

# 侧边栏输入
with st.sidebar:
    st.header("参数设置")
    stock_code = st.text_input('股票代码', '600000')
    start_date = st.date_input('开始日期', default_start)
    end_date = st.date_input('结束日期', default_end)
    train_ratio = st.slider('训练集比例', 0.6, 0.95, 0.8, 0.05)
    adjust_type = st.radio(
        "复权类型",
        options=[("前复权", "qfq"), ("后复权", "hfq"), ("不复权", '')],  # 显示文本与传值分离
        index=0,
        format_func=lambda x: x[0],  # 只显示文本部分
        help="前复权(qfq)/后复权(hfq)/不复权"
    )[1]


# 数据获取函数
@st.cache_data(ttl=3600, show_spinner="正在获取股票数据...")
def get_stock_data(code, start, end, adjust):
    try:
        # 自动处理代码前缀
        symbol =  code

        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust=adjust
        )

        # 数据清洗
        df = df.set_index('日期').sort_index()
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume'
        })
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        st.error(f"数据获取失败: {str(e)}")
        return pd.DataFrame()


# 获取数据
with st.spinner('正在加载数据...'):
    data = get_stock_data(stock_code, start_date, end_date, adjust_type)

if data.empty:
    st.error("无法获取数据，请检查：\n1. 股票代码是否正确\n2. 日期范围是否有效\n3. 网络连接是否正常")
    st.stop()


# 特征工程函数
def create_features(df):
    # 创建目标变量（次日是否上涨）
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.iloc[:-1]  # 删除最后一个无法确定target的交易日

    # 移动平均线
    windows = [5, 10, 20]
    for window in windows:
        df[f'ma{window}'] = df['close'].rolling(window).mean()

    # RSI计算
    delta = df['close'].diff().dropna()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # 防止除零错误
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD计算
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    return df.dropna()


# 处理数据
processed_data = create_features(data)

# 数据校验
if len(processed_data) < 100:
    st.warning(f"数据量不足（仅{len(processed_data)}条），建议选择更长的时间范围")
    st.stop()

# 特征列定义
features = ['open', 'high', 'low', 'close', 'volume',
            'ma5', 'ma10', 'ma20', 'rsi', 'macd', 'signal']

# 划分数据集
split_idx = int(len(processed_data) * train_ratio)
X = processed_data[features]
y = processed_data['target']

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# 模型训练
with st.spinner('正在训练模型...'):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 界面展示
col1, col2 = st.columns(2)

with col1:
    st.subheader('模型表现')
    st.metric("测试集准确率", f"{accuracy:.2%}")

    # 特征重要性
    st.subheader('特征重要性')
    importance = pd.DataFrame({
        '特征': features,
        '重要性': model.feature_importances_
    }).sort_values('重要性', ascending=False)
    st.bar_chart(importance.set_index('特征'))

with col2:
    st.subheader('价格走势与预测')

    # 创建K线图
    fig = go.Figure()

    # K线主图
    fig.add_trace(go.Candlestick(
        x=processed_data.index,
        open=processed_data['open'],
        high=processed_data['high'],
        low=processed_data['low'],
        close=processed_data['close'],
        name='K线'
    ))

    # 预测标记（测试集部分）
    test_dates = processed_data.index[split_idx:]
    predictions = pd.Series(y_pred, index=test_dates[:len(y_pred)])

    # 正确预测标记
    correct_dates = predictions[predictions == y_test].index
    fig.add_trace(go.Scatter(
        x=correct_dates,
        y=processed_data.loc[correct_dates, 'high'] * 1.02,
        mode='markers',
        marker=dict(color='lime', size=8, symbol='triangle-up'),
        name='正确预测'
    ))

    # 错误预测标记
    wrong_dates = predictions[predictions != y_test].index
    fig.add_trace(go.Scatter(
        x=wrong_dates,
        y=processed_data.loc[wrong_dates, 'low'] * 0.98,
        mode='markers',
        marker=dict(color='red', size=8, symbol='triangle-down'),
        name='错误预测'
    ))

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

# 风险提示
st.markdown("---")
st.warning("""
**风险提示**  
本工具仅为技术演示，不构成投资建议。股票市场存在风险，历史表现不代表未来趋势。实际投资请谨慎决策，作者不对任何投资结果负责。
""")

