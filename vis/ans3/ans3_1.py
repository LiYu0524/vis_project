import pandas as pd
import re
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# 读取CSV文件
df = pd.read_csv("data_rules.csv")

# 定义关键词列表，用于识别与死亡相关的消息
death_keywords = [
    # 核心
    "fatality", "fatalities", "death", "deaths", "dead", "died",
    "kill", "killed", "killing", "casualty", "casualties",
    # 扩展
    "victim", "victims", "deceased", "body", "bodies",
    "corpse", "corpses", "passed\\s+away", "perished",
    "lost\\s+(?:their|his|her)\\s+life", "lost\\s+(?:their|his|her)\\s+lives",
    "death\\s+toll", "body\\s+count", "counted\\s+dead", "pronounced\\s+dead",
    # 俚语 / 缩写
    "RIP", "R\\.I\\.P\\.", "rip", "no\\s+survivors",
    "gone", "didn't\\s+make\\s+it", "didnt\\s+make\\s+it", "🙏"
]
pattern = re.compile(r'\b(?:' + '|'.join(death_keywords) + r')\b', flags=re.IGNORECASE)

# 筛选出包含关键词的消息
death_related = df[df['cleanTweet'].str.contains(pattern, na=False)]

# 定义函数提取消息中出现的数字
def extract_fatality_numbers(text):
    numbers = re.findall(r'\b\d+\b', text)
    return [int(n) for n in numbers] if numbers else []

# 提取所有数字并展平
all_fatalities = []
for tweet in death_related['cleanTweet'].dropna():
    all_fatalities.extend(extract_fatality_numbers(tweet))

# 统计每个死亡人数的提及次数
fatality_counts = pd.Series(all_fatalities).value_counts().sort_index()

# # Matplotlib版本（已注释）
# plt.figure(figsize=(10, 6))
# plt.bar(fatality_counts.index, fatality_counts.values, color='orange')
# #plt.xscale('log')
# plt.xlabel("Number of reported fatalities in message")
# plt.ylabel("Number of mentions")
# plt.title("Reported Fatalities from Social Media Messages")
# plt.grid(True, which="both", linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.show()

# Plotly版本
fig = go.Figure()

# 添加柱状图
fig.add_trace(go.Bar(
    x=fatality_counts.index,
    y=fatality_counts.values,
    marker_color='orange',
    name='Fatality Mentions'
))

# 设置布局
fig.update_layout(
    title='Reported Fatalities from Social Media Messages',
    xaxis_title='Number of reported fatalities in message',
    yaxis_title='Number of mentions',
    width=1000,
    height=600,
    showlegend=False,
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True)
)

# 显示图表
fig.show()
fig.write_html('fatality_analysis.html')
fig.write_image('fatality_analysis.png', width=1000, height=600, scale=2)