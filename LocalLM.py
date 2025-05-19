import streamlit as st
import akshare as ak
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM
import requests
# 环境配置检查
def check_ollama_service():
    try:
        response = requests.get("http://localhost:11434", timeout=60)
        return response.status_code == 200
    except:
        return False
# 获取股票数据（带缓存）
@st.cache_data(ttl=3600)
def get_stock_data(stock_code):
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
    try:
        df = ak.stock_zh_a_hist(symbol=stock_code, period="daily",
                                start_date=start_date, end_date=end_date, adjust="qfq")
        return df[['日期', '开盘', '最高', '最低', '收盘', '成交量']]
    except Exception as e:
        st.error(f"数据获取失败: {str(e)}")
        return None
# 技术指标计算
def calculate_technical_indicators(df):
    # 计算移动平均线
    df['MA5'] = df['收盘'].rolling(5).mean()
    df['MA20'] = df['收盘'].rolling(20).mean()
    # 计算RSI
    delta = df['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    return df.dropna()
# AI分析模块
def analyze_with_ai(stock_code, df, news):
    template = """作为资深证券分析师，请基于以下数据给出专业分析：
    股票代码：{code}
    最新收盘价：{close}
    关键指标：
    - 5日均线：{ma5}
    - 20日均线：{ma20}
    - RSI指数：{rsi}
    近期新闻摘要：
    {news}
    请从以下维度给出分析：
    1. 趋势判断（技术面）
    2. 量价关系分析
    3. 新闻事件影响评估
    4. 短期操作建议
    5. 风险提示"""
    llm = OllamaLLM(
        model="qwen3:4b",
        temperature=0.7,
        top_p=0.9,
        num_ctx=2048,
        num_gpu=1
    )
    analysis = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(template)
    ).run({
        "code": stock_code,
        "close": df['收盘'].iloc[-1],
        "ma5": df['MA5'].iloc[-1],
        "ma20": df['MA20'].iloc[-1],
        "rsi": df['RSI'].iloc[-1],
        "news": "\n".join(news['新闻标题'].head(3))
    })
    print(analysis)
    return analysis
# 主程序
def main():
    st.title("智能股票分析系统")
    if not check_ollama_service():
        st.error("Ollama服务未启动！请执行：ollama serve")
        return
    stock_code = st.text_input('输入股票代码（例如：600000）', '600519')
    if stock_code:
        # 数据获取模块
        df = get_stock_data(stock_code)
        if df is not None:
            df = calculate_technical_indicators(df)
            # K线图绘制
            fig = go.Figure(data=[
                go.Candlestick(x=df['日期'], open=df['开盘'],
                               high=df['最高'], low=df['最低'], close=df['收盘'],
                               increasing_line_color='red', decreasing_line_color='green'),
                go.Scatter(x=df['日期'], y=df['MA5'], name='5日均线'),
                go.Scatter(x=df['日期'], y=df['MA20'], name='20日均线')
            ])
            fig.update_layout(title=f'{stock_code} 技术分析', height=600)
            st.plotly_chart(fig, use_container_width=True)
            # 新闻获取模块
            try:
                news = ak.stock_news_em(symbol=stock_code).head(5)
                st.subheader("最新市场动态")
                for _, row in news.iterrows():
                    st.markdown(f"**[{row['新闻标题']}]({row['新闻链接']})** - {row['发布时间']}")
            except Exception as e:
                st.warning(f"新闻获取异常：{str(e)}")
            # AI分析模块
            with st.spinner("AI分析中..."):
                print(news)
                analysis = analyze_with_ai(stock_code, df, news)
                st.subheader("量化策略建议")
                st.markdown(f"```\n{analysis}\n```")
    st.caption("免责声明：本工具仅提供数据分析参考，不构成投资建议")
if __name__ == "__main__":
    main()

 




