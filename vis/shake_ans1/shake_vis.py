import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('num_with_shake.csv')

# 将时间列转换为datetime格式
df['time'] = pd.to_datetime(df['time'])
x = df['time']
y = df['shake_num']

# 找出所有波峰，设置最小间距参数避免过于密集的波峰
# distance参数表示波峰间的最小距离（以数据点为单位）
# 这里设置为12，表示波峰间至少间隔12个小时（12个数据点）
peaks, properties = find_peaks(df['shake_num'], prominence=10, distance=12)

fig, ax = plt.subplots(figsize=(12, 6))

# 添加阴影块（以峰值前后 1.5 小时为地震区间）
for idx in peaks:
    quake_time = df['time'].iloc[idx]
    ax.axvspan(quake_time - pd.Timedelta(hours=1.5), 
               quake_time + pd.Timedelta(hours=1.5), 
               color='orange', alpha=0.2, label='Earthquake Interval' if idx == peaks[0] else "")

# 绘制主线图
ax.plot(x, y, color='#00BFC4', linewidth=2, label='shake_num')

# 标记波峰点
ax.scatter(df['time'].iloc[peaks], df['shake_num'].iloc[peaks], 
           color='#7CAE00', s=50, zorder=5, label='Detected Peaks')

# 在x轴上额外标记波峰时间点
peak_times = df['time'].iloc[peaks]
for peak_time in peak_times:
    ax.axvline(x=peak_time, color='#7CAE00', linestyle='--', alpha=0.5, linewidth=1)

# 设置x轴显示为日期格式，包含波峰时间点
all_times = list(peak_times) + list(pd.date_range(start=df['time'].min(), end=df['time'].max(), freq='12H'))
all_times = sorted(set(all_times))
ax.set_xticks(all_times)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

plt.xticks(rotation=45)
plt.xlabel('time')
plt.ylabel('shake_num')
plt.title('Shake Events')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('shake_vis.png')