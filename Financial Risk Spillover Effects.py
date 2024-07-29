import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import adfuller
from statsmodels.tsa.api import VAR
import mindspore as ms
from mindspore import Tensor
import matplotlib.pyplot as plt

# 读取Excel数据
data_path = r'C:\Users\86158\Desktop\数据.xlsx'
df = pd.read_excel(data_path)
data = df.iloc[1:, 1:].values
dates = pd.to_datetime(df.iloc[1:, 0].values)

# 去除周末数据
df['is_weekend'] = df.iloc[1:, 0].apply(lambda x: x.weekday() >= 5)
df = df[df['is_weekend'] == False]
data = df.iloc[1:, 1:].values
dates = pd.to_datetime(df.iloc[1:, 0].values)

# 计算收益率
returns = np.diff(np.log(data), axis=0) * 100

# ADF检验
adf_results = [adfuller(returns[:, i])[1] for i in range(returns.shape[1])]

# VAR模型估计和BIC计算
model_order = 5
bic = []
for i in range(1, model_order + 1):
    model = VAR(returns)
    result = model.fit(i)
    bic.append(result.bic)

best_order = np.argmin(bic) + 1
var_model = VAR(returns)
var_result = var_model.fit(best_order)

# 检验特征根是否稳定
eigvals = np.linalg.eigvals(var_result.coefs)
root_test = [np.abs(val) <= 1 for val in eigvals]

# 计算FEVD
fevd_result = var_result.fevd(5)
gfevd5 = fevd_result.decomp[:, :, 4]

ngfevd5 = gfevd5 / gfevd5.sum(axis=0, keepdims=True)
fromto = np.zeros((ngfevd5.shape[0], 3))
for i in range(ngfevd5.shape[0]):
    fromto[i, 0] = ngfevd5[:, i].sum() - ngfevd5[i, i]
    fromto[i, 1] = ngfevd5[i, :].sum() - ngfevd5[i, i]
    fromto[i, 2] = fromto[i, 0] - fromto[i, 1]

fevd5_total_sum = (ngfevd5.sum() - np.trace(ngfevd5)) / 3

# 创建RateGroup
rate_group = [returns[i:i + 200, :] for i in range(len(returns) - 199)]

# VAR模型参数估计
ar_cells = []
cov_cells = []
for rg in rate_group:
    model = VAR(rg)
    result = model.fit(best_order)
    ar_cells.append(result.coefs)
    cov_cells.append(result.sigma_u)

# 计算NgFEVD5
ngfevd5_sum = []
ngfevd5_cells = []
for ar, cov in zip(ar_cells, cov_cells):
    result = VAR(returns).fit(maxlags=best_order)
    result.coefs = ar
    result.sigma_u = cov
    fevd_result = result.fevd(5)
    gfevd5 = fevd_result.decomp[:, :, 4]
    ngfevd5 = gfevd5 / gfevd5.sum(axis=0, keepdims=True)
    ngfevd5_sum.append((ngfevd5.sum() - np.trace(ngfevd5)) / 3)
    ngfevd5_cells.append(ngfevd5)

# 绘图
out = np.zeros((len(ngfevd5_cells), 6))
for i, ngfevd5 in enumerate(ngfevd5_cells):
    out[i, 0] = ngfevd5[0, 1]
    out[i, 1] = ngfevd5[0, 2]
    out[i, 2] = ngfevd5[1, 0]
    out[i, 3] = ngfevd5[1, 2]
    out[i, 4] = ngfevd5[2, 0]
    out[i, 5] = ngfevd5[2, 1]

t1 = dates[200:]
plt.plot(t1, ngfevd5_sum)
plt.xlim([min(t1), max(t1)])
plt.xlabel('Date')
plt.ylabel('NgFEVD5 Sum')
plt.legend(['TSI'])
plt.show()

# FromTo, ToSeries, FromSeries, NetSeries计算
from_series = []
to_series = []
net_series = []
for ngfevd5 in ngfevd5_cells:
    for i in range(ngfevd5.shape[0]):
        fromto[i, 0] = ngfevd5[:, i].sum() - ngfevd5[i, i]
        fromto[i, 1] = ngfevd5[i, :].sum() - ngfevd5[i, i]
        fromto[i, 2] = fromto[i, 0] - fromto[i, 1]
    to_series.append(fromto[:, 0])
    from_series.append(fromto[:, 1])
    net_series.append(fromto[:, 2])

# 绘制ToSeries, FromSeries, NetSeries
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(t1, np.array(to_series)[:, i] / 100, '-r', label='To')
    ax.plot(t1, np.array(from_series)[:, i] / 100, '-.g', label='From')
    ax.plot(t1, np.array(net_series)[:, i] / 100, '--b', label='Net')
    ax.axhline(0, color='k', linestyle=':')
    ax.set_xlim([min(t1), max(t1)])
    ax.set_title(f'Series {i+1}')
    ax.legend()

plt.xlabel('Date')
plt.show()
