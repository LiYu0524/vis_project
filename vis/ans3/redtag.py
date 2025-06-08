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

df = pd.read_csv('redtag.csv')

# 将时间列转换为datetime格式
df['time'] = pd.to_datetime(df['time'])
x = df['time']
y = df['count']

# 设置阈值过滤较低的波峰
height_threshold = 5  # 只保留高度大于5的波峰
# 找出所有波峰，设置最小间距参数避免过于密集的波峰
# distance参数表示波峰间的最小距离（以数据点为单位）
# 这里设置为4，表示波峰间至少间隔4个小时（4个数据点）
peaks, properties = find_peaks(df['count'], prominence=3, distance=4, height=height_threshold)

# Plotly版本
fig = go.Figure()

# 添加主线图
fig.add_trace(go.Scatter(
    x=x, y=y,
    mode='lines',
    name='redtag_count',
    line=dict(color='#DC143C', width=2)
))

# 标记波峰点
if len(peaks) > 0:
    fig.add_trace(go.Scatter(
        x=df['time'].iloc[peaks],
        y=df['count'].iloc[peaks],
        mode='markers',
        name='Detected Peaks',
        marker=dict(color='#8B0000', size=12)
    ))

# 添加阴影块和垂直线（以峰值前后 2 小时为红标高峰区间）
for idx in peaks:
    peak_time = df['time'].iloc[idx]
    start_time = peak_time - pd.Timedelta(hours=2)
    end_time = peak_time + pd.Timedelta(hours=2)
    
    # 添加阴影块
    fig.add_vrect(
        x0=start_time, x1=end_time,
        fillcolor="lightcoral", opacity=0.25,
        layer="below", line_width=0,
        annotation_text="High Redtag Interval" if idx == peaks[0] else "",
        annotation_position="top left"
    )
    
    # 添加垂直线标记波峰时间点
    fig.add_vline(
        x=peak_time,
        line=dict(color='#DC143C', width=2, dash='dashdot'),
        opacity=0.6
    )

# 设置布局
fig.update_layout(
    title='Redtag Events',
    xaxis_title='time',
    yaxis_title='redtag_count',
    width=1200,
    height=600,
    showlegend=True
)

# 显示图表
fig.show()
fig.write_html('redtag_vis.html')
fig.write_image('redtag_vis.png', width=1200, height=600, scale=2)
