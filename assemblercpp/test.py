import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.special import erf

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號無法顯示的問題
# 常態分佈的密度函數（PDF）
def normal(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# 累積分佈函數（CDF）
def cdf(x, mu, sigma):
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))  # 使用scipy的erf

# 畫圖函數
def plot_distribution(mu, sigma):
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)  # 定義x範圍
    y_pdf = normal(x, mu, sigma)  # 計算PDF
    y_cdf = cdf(x, mu, sigma)  # 計算CDF
    plt.figure(figsize=(6,5))

    # 畫PDF
    plt.plot(x, y_pdf, label='概率密度函數 (PDF)', color='b')
    plt.title(f'常態分佈的 PDF 和 CDF\n(均值={mu}, 標準差={sigma})')
    plt.xlabel('x')
    plt.ylabel('密度')

    # 畫CDF
    plt.plot(x, y_cdf, label='累積分佈函數 (CDF)', color='r')

    # 顯示圖例
    plt.legend()

    plt.tight_layout()
    plt.show()

# 輸入參數
mu = 1  # 均值
sigma = 1  # 標準差

plot_distribution(mu, sigma)
