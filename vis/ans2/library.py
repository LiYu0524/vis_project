import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('library_crash_count.csv')

# 将时间列转换为datetime格式
df['time'] = pd.to_datetime(df['time'])
x = df['time']
y = df['count']

# 设置阈值过滤较低的波峰
height_threshold = 20  # 只保留高度大于10的波峰
# 找出所有波峰，设置最小间距参数避免过于密集的波峰
# distance参数表示波峰间的最小距离（以数据点为单位）
# 这里设置为6，表示波峰间至少间隔6个小时（6个数据点）
peaks, properties = find_peaks(df['count'], prominence=5, distance=6, height=height_threshold)

# # Matplotlib版本（已注释）
# fig, ax = plt.subplots(figsize=(12, 6))
# 
# # 添加阴影块（以峰值前后 2 小时为崩溃高峰区间）
# for idx in peaks:
#     crash_time = df['time'].iloc[idx]
#     ax.axvspan(crash_time - pd.Timedelta(hours=2), 
#                crash_time + pd.Timedelta(hours=2), 
#                color='red', alpha=0.15, label='High Crash Interval' if idx == peaks[0] else "")
# 
# # 绘制主线图
# ax.plot(x, y, color='#E31A1C', linewidth=2, label='crash_count')
# 
# # 标记波峰点
# ax.scatter(df['time'].iloc[peaks], df['count'].iloc[peaks], 
#            color='#FF7F00', s=60, zorder=5, label='Detected Peaks')
# 
# # 在x轴上额外标记波峰时间点
# peak_times = df['time'].iloc[peaks]
# for peak_time in peak_times:
#     ax.axvline(x=peak_time, color='#FF7F00', linestyle=':', alpha=0.7, linewidth=1.5)
# 
# # 设置x轴显示为日期格式，包含波峰时间点
# all_times = list(peak_times) + list(pd.date_range(start=df['time'].min(), end=df['time'].max(), freq='12H'))
# all_times = sorted(set(all_times))
# ax.set_xticks(all_times)
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
# 
# plt.xticks(rotation=45)
# plt.xlabel('time')
# plt.ylabel('crash_count')
# plt.title('Library Crash Events')
# plt.legend()
# plt.tight_layout()
# plt.show()
# plt.savefig('library_crash_vis.png')

# Plotly版本
fig = go.Figure()

# 添加主线图
fig.add_trace(go.Scatter(
    x=x, y=y,
    mode='lines',
    name='crash_count',
    line=dict(color='#E31A1C', width=2)
))

# 标记波峰点
if len(peaks) > 0:
    fig.add_trace(go.Scatter(
        x=df['time'].iloc[peaks],
        y=df['count'].iloc[peaks],
        mode='markers',
        name='libary crash point ',
        marker=dict(color='#FF7F00', size=10)
    ))

# 添加阴影块和垂直线（以峰值前后 2 小时为崩溃高峰区间）
for idx in peaks:
    crash_time = df['time'].iloc[idx]
    start_time = crash_time - pd.Timedelta(hours=2)
    end_time = crash_time + pd.Timedelta(hours=2)
    
    # 添加阴影块
    fig.add_vrect(
        x0=start_time, x1=end_time,
        fillcolor="red", opacity=0.15,
        layer="below", line_width=0,
        annotation_text="High Crash Interval" if idx == peaks[0] else "",
        annotation_position="top left"
    )
    
    # 添加垂直线标记波峰时间点
    fig.add_vline(
        x=crash_time,
        line=dict(color='#FF7F00', width=1.5, dash='dot'),
        opacity=0.7
    )

# 设置布局
fig.update_layout(
    title='Library Crash Events',
    xaxis_title='time',
    yaxis_title='crash_count',
    width=1200,
    height=600,
    showlegend=True
)

# 显示图表
fig.show()
fig.write_html('library_crash_vis.html')
fig.write_image('library_crash_vis.png', width=1200, height=600, scale=2)
