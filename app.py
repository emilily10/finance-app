import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
from IPython.display import Markdown

# ==========================================
# 1. 基础配置与数据下载 (解决 KeyError)
# ==========================================
tickers = ['AAPL', 'TSLA', 'GLD', 'MSFT', 'BTC-USD'] 
investment = 10000 
risk_free_rate = 0.02 # 假设无风险利率

print("正在获取实时市场数据...")
raw_data = yf.download(tickers, period="2y")

# 核心修正：自动处理 yfinance 返回的多级索引
if isinstance(raw_data.columns, pd.MultiIndex):
    # 优先选 Adj Close，没有就选 Close
    price_col = 'Adj Close' if 'Adj Close' in raw_data.columns.levels[0] else 'Close'
    data = raw_data[price_col]
else:
    data = raw_data[['Adj Close']] if 'Adj Close' in raw_data.columns else raw_data[['Close']]

# 清洗数据
returns = data.pct_change().dropna()
mean_rets = returns.mean()
cov_matrix = returns.cov()

# ==========================================
# 2. 核心算法：收益最大化 (解决变量定义问题)
# ==========================================
def portfolio_stats(weights):
    p_ret = np.sum(mean_rets * weights) * 252
    p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return p_ret, p_std

# 追求夏普比率最大化（性价比最高的高收益组合）
def negative_sharpe(weights):
    p_ret, p_std = portfolio_stats(weights)
    return -(p_ret - risk_free_rate) / p_std

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(len(tickers)))
init_guess = [1./len(tickers)] * len(tickers)

opt_results = minimize(negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = opt_results.x

# 在这里明确定义 ann_vol，防止下游出现 NameError
ann_ret, ann_vol = portfolio_stats(optimal_weights)

print("\n" + "="*30)
print("🎯 收益最大化配置结果:")
for ticker, weight in zip(data.columns, optimal_weights):
    print(f"{ticker}: {weight:.2%}")
print(f"预期年化收益: {ann_ret:.2%}")
print(f"预期年化风险 (波动率): {ann_vol:.2%}")
print("="*30 + "\n")

# ==========================================
# 4. 一键模拟：黑天鹅压力测试 (Black Swan)
# ==========================================
# 注入危机因子：模拟极端年份收益跌 40%，波动率翻倍
crash_ret = -0.40 
crash_vol = ann_vol * 2 # 这里已经保证 ann_vol 被定义了
n_sims, n_days = 100, 252

np.random.seed(42)
sim_rets = np.random.normal(crash_ret/n_days, crash_vol/np.sqrt(n_days), (n_days, n_sims))
sim_paths = investment * (1 + sim_rets).cumprod(axis=0)

# 画图
fig = go.Figure()
for i in range(15): # 画出部分随机路径
    fig.add_trace(go.Scatter(y=sim_paths[:, i], mode='lines', line=dict(width=0.5), opacity=0.3, showlegend=False))

mean_path = sim_paths.mean(axis=1)
fig.add_trace(go.Scatter(y=mean_path, mode='lines', name='平均危机走势', line=dict(color='red', width=4)))

fig.update_layout(title="🔥 一键黑天鹅压力测试", template="plotly_dark", 
                  xaxis_title="交易日", yaxis_title="账户价值 ($)")
fig.show()

worst_case = sim_paths[-1, :].min()
print(f"🚨 [黑天鹅警报]: 极端情况下资产可能缩水至 ${worst_case:,.2f}")
