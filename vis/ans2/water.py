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

df = pd.read_csv('water_issue_count.csv')

# 将时间列转换为datetime格式
df['time'] = pd.to_datetime(df['time'])
x = df['time']
y = df['count']

# 设置阈值过滤较低的波峰
height_threshold = 100  # 只保留高度大于30的波峰
# 找出所有波峰，设置最小间距参数避免过于密集的波峰
# distance参数表示波峰间的最小距离（以数据点为单位）
# 这里设置为8，表示波峰间至少间隔8个小时（8个数据点）
peaks, properties = find_peaks(df['count'], prominence=15, distance=8, height=height_threshold)

# # Matplotlib版本（已注释）
# fig, ax = plt.subplots(figsize=(12, 6))
# 
# # 添加阴影块（以峰值前后 3 小时为水问题高峰区间）
# for idx in peaks:
#     issue_time = df['time'].iloc[idx]
#     ax.axvspan(issue_time - pd.Timedelta(hours=3), 
#                issue_time + pd.Timedelta(hours=3), 
#                color='lightblue', alpha=0.25, label='High Water Issue Interval' if idx == peaks[0] else "")
# 
# # 绘制主线图
# ax.plot(x, y, color='#1F78B4', linewidth=2, label='water_issue_count')
# 
# # 标记波峰点
# ax.scatter(df['time'].iloc[peaks], df['count'].iloc[peaks], 
#            color='#33A02C', s=70, zorder=5, label='Detected Peaks')
# 
# # 在x轴上额外标记波峰时间点
# peak_times = df['time'].iloc[peaks]
# for peak_time in peak_times:
#     ax.axvline(x=peak_time, color='#33A02C', linestyle='-.', alpha=0.6, linewidth=2)
# 
# # 设置x轴显示为日期格式，包含波峰时间点
# all_times = list(peak_times) + list(pd.date_range(start=df['time'].min(), end=df['time'].max(), freq='12H'))
# all_times = sorted(set(all_times))
# ax.set_xticks(all_times)
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
# 
# plt.xticks(rotation=45)
# plt.xlabel('time')
# plt.ylabel('water_issue_count')
# plt.title('Water Issue Events')
# plt.legend()
# plt.tight_layout()
# plt.show()
# plt.savefig('water_issue_vis.png')

# Plotly版本
fig = go.Figure()

# 添加主线图
fig.add_trace(go.Scatter(
    x=x, y=y,
    mode='lines',
    name='water_issue_count',
    line=dict(color='#1F78B4', width=2)
))

# 标记波峰点
if len(peaks) > 0:
    fig.add_trace(go.Scatter(
        x=df['time'].iloc[peaks],
        y=df['count'].iloc[peaks],
        mode='markers',
        name='water issue spike',
        marker=dict(color='#33A02C', size=12)
    ))

# 添加阴影块和垂直线（以峰值前后 3 小时为水问题高峰区间）
for idx in peaks:
    issue_time = df['time'].iloc[idx]
    start_time = issue_time - pd.Timedelta(hours=3)
    end_time = issue_time + pd.Timedelta(hours=3)
    
    # 添加阴影块
    fig.add_vrect(
        x0=start_time, x1=end_time,
        fillcolor="lightblue", opacity=0.25,
        layer="below", line_width=0,
        annotation_text="High Water Issue Interval" if idx == peaks[0] else "",
        annotation_position="top left"
    )
    
    # 添加垂直线标记波峰时间点
    fig.add_vline(
        x=issue_time,
        line=dict(color='#33A02C', width=2, dash='dashdot'),
        opacity=0.6
    )

# 设置布局
fig.update_layout(
    title='Water Issue Events',
    xaxis_title='time',
    yaxis_title='water_issue_count',
    width=1200,
    height=600,
    showlegend=True
)

# 显示图表
fig.show()
fig.write_html('water_issue_vis.html')
fig.write_image('water_issue_vis.png', width=1200, height=600, scale=2)
