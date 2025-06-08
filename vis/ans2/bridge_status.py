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

df = pd.read_csv('bridge_status.csv')

# 将时间列转换为datetime格式
df['time'] = pd.to_datetime(df['time'])

# 过滤数据：只保留至少有一个值不为0的时间点
df_filtered = df[(df['open'] != 0) | (df['close'] != 0) | (df['uncertain'] != 0)].copy()

x = df_filtered['time']
y_open = df_filtered['open']
y_close = df_filtered['close']
y_uncertain = df_filtered['uncertain']

# 设置阈值过滤较低的波峰（只对open数据）
height_threshold = 31  # 只保留open值大于10的波峰
# 找出open数据的波峰，设置最小间距参数避免过于密集的波峰
# distance参数表示波峰间的最小距离（以数据点为单位）
# 这里设置为4，表示波峰间至少间隔4个小时（4个数据点）
# 降低prominence以检测更多波峰
peaks, properties = find_peaks(df_filtered['open'], prominence=8, distance=4, height=height_threshold)

# Plotly版本
fig = go.Figure()

# 添加close线条（浅灰色）
fig.add_trace(go.Scatter(
    x=x, y=y_close,
    mode='lines',
    name='close',
    line=dict(color='#CCCCCC', width=2)
))

# 添加uncertain线条（深灰色）
fig.add_trace(go.Scatter(
    x=x, y=y_uncertain,
    mode='lines',
    name='uncertain',
    line=dict(color='#999999', width=2)
))

# 添加open线条（绿色）
fig.add_trace(go.Scatter(
    x=x, y=y_open,
    mode='lines',
    name='open',
    line=dict(color='#00AA00', width=2)
))

# 标记open的波峰点
if len(peaks) > 0:
    fig.add_trace(go.Scatter(
        x=df_filtered['time'].iloc[peaks],
        y=df_filtered['open'].iloc[peaks],
        mode='markers',
        name='Open Peaks',
        marker=dict(color='#006600', size=12)
    ))

# 添加阴影块和垂直线（以open波峰前后 2 小时为高峰区间）
for idx in peaks:
    peak_time = df_filtered['time'].iloc[idx]
    start_time = peak_time - pd.Timedelta(hours=2)
    end_time = peak_time + pd.Timedelta(hours=2)
    
    # 添加阴影块（浅绿色）
    fig.add_vrect(
        x0=start_time, x1=end_time,
        fillcolor="lightgreen", opacity=0.25,
        layer="below", line_width=0,
        annotation_text="High Open Interval" if idx == peaks[0] else "",
        annotation_position="top left"
    )
    
    # 添加垂直线标记波峰时间点
    fig.add_vline(
        x=peak_time,
        line=dict(color='#00AA00', width=2, dash='dashdot'),
        opacity=0.6
    )

# 设置布局
fig.update_layout(
    title='Bridge Status Events',
    xaxis_title='time',
    yaxis_title='status_count',
    width=1200,
    height=600,
    showlegend=True
)

# 显示图表
fig.show()
fig.write_html('bridge_status_vis.html')
fig.write_image('bridge_status_vis.png', width=1200, height=600, scale=2)