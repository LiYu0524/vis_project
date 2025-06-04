import json
import pandas as pd

# 读取 GeoJSON 文件
with open("st_himark.geojson", "r") as f:
    geojson_data = json.load(f)

# 读取CSV数据
df = pd.read_csv("min_sentiment_score_for_all_locations_area_wgt.csv")

# 数据预处理
df['decision_time'] = pd.to_datetime(df['decision_time'])
df = df.sort_values(['decision_time', 'location'])

# 获取所有唯一的时间点和地区
unique_times = sorted(df['decision_time'].unique())
unique_locations = sorted(df['location'].unique())

print(f"数据时间范围: {unique_times[0]} 到 {unique_times[-1]}")
print(f"共 {len(unique_times)} 个时间点")
print(f"地区数量: {len(unique_locations)}")

# 使用所有时间点（每小时）
print(f"使用全部 {len(unique_times)} 个时间点")

# 计算area adjusted列的全局最大最小值
area_adjusted_columns = [
    'building damage issues area adjusted',
    'medical issues area adjusted', 
    'power issues area adjusted',
    'road issues area adjusted',
    'water issues area adjusted',
    'min_sentiment_score area adjusted'
]

# 计算全局范围
global_ranges = {}
for col in area_adjusted_columns:
    valid_data = df[col].dropna()
    if len(valid_data) > 0:
        global_ranges[col] = {
            'min': float(valid_data.min()),
            'max': float(valid_data.max())
        }
        print(f"{col}: 范围 {global_ranges[col]['min']:.4f} - {global_ranges[col]['max']:.4f}")
    else:
        global_ranges[col] = {'min': 0, 'max': 1}
        print(f"{col}: 无有效数据，使用默认范围 0-1")

# 准备所有时间点的数据
all_data = {}
for i, time_point in enumerate(unique_times):
    if i % 20 == 0:
        print(f"处理数据: {i+1}/{len(unique_times)} - {time_point}")
    
    time_key = time_point.strftime('%Y%m%d%H%M')
    current_data = df[df['decision_time'] == time_point]
    
    # 为每种数据类型准备数据
    data_types = {
        'min_sentiment_score': 'min_sentiment_score',
        'building_damage': 'building damage issues area adjusted',
        'medical': 'medical issues area adjusted', 
        'power': 'power issues area adjusted',
        'road': 'road issues area adjusted',
        'water': 'water issues area adjusted',
        'min_sentiment_area': 'min_sentiment_score area adjusted'
    }
    
    type_data = {}
    for data_key, column_name in data_types.items():
        map_data = []
        for location in unique_locations:
            location_data = current_data[current_data['location'] == location]
            
            if not location_data.empty and pd.notna(location_data.iloc[0][column_name]):
                score = float(location_data.iloc[0][column_name])
                issues = location_data.iloc[0]['min_sentiment_issues'] if column_name.startswith('min_sentiment') else column_name
                
                # 根据列类型选择不同的处理方式
                if column_name in area_adjusted_columns:
                    # area adjusted列：使用动态缩放
                    col_range = global_ranges[column_name]
                    if col_range['max'] > col_range['min']:
                        # 线性缩放到0-1范围
                        scaled_value = (score - col_range['min']) / (col_range['max'] - col_range['min'])
                    else:
                        scaled_value = 0.5  # 如果max=min，使用中值
                    # 反转处理
                    display_score = (1 - scaled_value) * 100
                else:
                    # 原始列：使用原有的0-100处理
                    display_score = (1 - score) * 100
                
                time_str = time_point.strftime('%Y-%m-%d %H:%M:%S')
                issues_str = issues if pd.notna(issues) else 'N/A'
                
                map_data.append({
                    "name": location,
                    "value": display_score,
                    "tooltip": f"{location}<br/>严重程度: {display_score:.1f}<br/>时间: {time_str}<br/>问题类型: {issues_str}"
                })
            else:
                time_str = time_point.strftime('%Y-%m-%d %H:%M:%S')
                map_data.append({
                    "name": location,
                    "value": None,  # 使用None表示无数据，触发灰色
                    "tooltip": f"{location}<br/>时间: {time_str}<br/>状态: 无数据"
                })
        
        type_data[data_key] = map_data
    
    all_data[time_key] = {
        "time_display": time_point.strftime('%Y-%m-%d %H:%M:%S'),
        "time_short": time_point.strftime('%m-%d %H:%M'),
        "data": type_data
    }

print("开始生成HTML文件...")

# 准备色阶范围信息
scale_ranges = {}
for data_key, column_name in data_types.items():
    if column_name in area_adjusted_columns:
        col_range = global_ranges[column_name]
        scale_ranges[data_key] = {
            'min': 0,
            'max': 100,
            'original_min': col_range['min'],
            'original_max': col_range['max'],
            'type': 'area_adjusted'
        }
    else:
        scale_ranges[data_key] = {
            'min': 0,
            'max': 100,
            'original_min': 0,
            'original_max': 1,
            'type': 'original'
        }

# 生成HTML文件
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>St. Himark 时间线地图</title>
    <script src="https://assets.pyecharts.org/assets/v5/echarts.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 28px;
            font-weight: 300;
        }}
        .header p {{
            margin: 0;
            opacity: 0.9;
            font-size: 16px;
        }}
        #map-container {{
            width: 100%;
            height: 700px;
            background-color: #f8f9fa;
        }}
        .controls {{
            padding: 25px;
            background: linear-gradient(135deg, #ecf0f1 0%, #d5dbdb 100%);
            border-top: 1px solid #bdc3c7;
        }}
        .data-type-selector {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .data-type-label {{
            font-size: 18px;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .data-type-buttons {{
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .data-btn {{
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 8px 16px;
            margin: 4px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
            white-space: nowrap;
        }}
        .data-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
        }}
        .data-btn:active {{
            transform: translateY(0);
        }}
        .data-btn:disabled {{
            background: #95a5a6;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }}
        .data-btn.active {{
            background: linear-gradient(135deg, #2ecc71, #27ae60);
        }}
        .time-display {{
            text-align: center;
            font-size: 20px;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 20px;
            padding: 10px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .slider-container {{
            margin: 20px 0;
            position: relative;
        }}
        .time-slider {{
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: #ddd;
            outline: none;
            -webkit-appearance: none;
            appearance: none;
        }}
        .time-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: linear-gradient(135deg, #3498db, #2980b9);
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        .time-slider::-moz-range-thumb {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: linear-gradient(135deg, #3498db, #2980b9);
            cursor: pointer;
            border: none;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        .control-buttons {{
            text-align: center;
            margin: 20px 0;
        }}
        .btn {{
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 12px 24px;
            margin: 0 8px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        }}
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
        }}
        .btn:active {{
            transform: translateY(0);
        }}
        .btn:disabled {{
            background: #95a5a6;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }}
        .progress-info {{
            text-align: center;
            margin-top: 15px;
            color: #7f8c8d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>St. Himark 区域严重程度时间线地图</h1>
            <p>数据时间范围: {unique_times[0].strftime('%Y-%m-%d %H:%M')} 到 {unique_times[-1].strftime('%Y-%m-%d %H:%M')} | 共 {len(unique_times)} 个时间点</p>
        </div>
        
        <div id="map-container"></div>
        
        <div class="controls">
            <div class="data-type-selector">
                <div class="data-type-label">当前数据类型: <span id="current-data-type">最小情感分数</span></div>
                <div class="data-type-buttons">
                    <button class="data-btn active" data-type="min_sentiment_score">最小情感分数</button>
                    <button class="data-btn" data-type="building_damage">建筑损坏(面积调整)</button>
                    <button class="data-btn" data-type="medical">医疗问题(面积调整)</button>
                    <button class="data-btn" data-type="power">电力问题(面积调整)</button>
                    <button class="data-btn" data-type="road">道路问题(面积调整)</button>
                    <button class="data-btn" data-type="water">供水问题(面积调整)</button>
                    <button class="data-btn" data-type="min_sentiment_area">最小情感分数(面积调整)</button>
                </div>
            </div>
            
            <div class="time-display" id="current-time">
                {unique_times[0].strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            
            <div class="slider-container">
                <input type="range" 
                       class="time-slider" 
                       id="time-slider" 
                       min="0" 
                       max="{len(unique_times)-1}" 
                       value="0" 
                       step="1">
            </div>
            
            <div class="control-buttons">
                <button class="btn" id="prev-btn">⏮ 上一个</button>
                <button class="btn" id="play-btn">▶ 播放</button>
                <button class="btn" id="next-btn">⏭ 下一个</button>
                <button class="btn" id="reset-btn">🔄 重置</button>
            </div>
            
            <div class="progress-info" id="progress-info">
                第 1 个时间点，共 {len(unique_times)} 个
            </div>
        </div>
    </div>

    <script>
        // 地图数据
        const geoJsonData = {json.dumps(geojson_data)};
        const timelineData = {json.dumps(all_data)};
        const timeKeys = {json.dumps(list(all_data.keys()))};
        const scaleRanges = {json.dumps(scale_ranges)};
        
        // 初始化ECharts
        const chart = echarts.init(document.getElementById('map-container'));
        
        // 注册地图
        echarts.registerMap('StHimark', geoJsonData);
        
        // 当前时间索引
        let currentTimeIndex = 0;
        let isPlaying = false;
        let playInterval = null;
        
        // 当前数据类型
        let currentDataType = 'min_sentiment_score';
        
        // 数据类型名称映射（包含原始范围信息）
        const dataTypeNames = {{
            'min_sentiment_score': '最小情感分数',
            'building_damage': '建筑损坏(面积调整) [' + scaleRanges.building_damage.original_min.toFixed(3) + '-' + scaleRanges.building_damage.original_max.toFixed(3) + ']',
            'medical': '医疗问题(面积调整) [' + scaleRanges.medical.original_min.toFixed(3) + '-' + scaleRanges.medical.original_max.toFixed(3) + ']',
            'power': '电力问题(面积调整) [' + scaleRanges.power.original_min.toFixed(3) + '-' + scaleRanges.power.original_max.toFixed(3) + ']',
            'road': '道路问题(面积调整) [' + scaleRanges.road.original_min.toFixed(3) + '-' + scaleRanges.road.original_max.toFixed(3) + ']',
            'water': '供水问题(面积调整) [' + scaleRanges.water.original_min.toFixed(3) + '-' + scaleRanges.water.original_max.toFixed(3) + ']',
            'min_sentiment_area': '最小情感分数(面积调整) [' + scaleRanges.min_sentiment_area.original_min.toFixed(3) + '-' + scaleRanges.min_sentiment_area.original_max.toFixed(3) + ']'
        }};
        
        // DOM元素
        const timeSlider = document.getElementById('time-slider');
        const currentTimeDisplay = document.getElementById('current-time');
        const progressInfo = document.getElementById('progress-info');
        const playBtn = document.getElementById('play-btn');
        const prevBtn = document.getElementById('prev-btn');
        const nextBtn = document.getElementById('next-btn');
        const resetBtn = document.getElementById('reset-btn');
        const currentDataTypeDisplay = document.getElementById('current-data-type');
        const dataTypeBtns = document.querySelectorAll('.data-btn');
        
        // 更新地图
        function updateMap(timeIndex) {{
            const timeKey = timeKeys[timeIndex];
            const timeData = timelineData[timeKey];
            const mapData = timeData.data[currentDataType];
            const currentRange = scaleRanges[currentDataType];
            
            // 根据数据类型确定色阶标签
            let scaleText = ['高风险', '低风险'];
            if (currentRange.type === 'area_adjusted') {{
                scaleText = [
                    '高风险 (' + currentRange.original_max.toFixed(3) + ')',
                    '低风险 (' + currentRange.original_min.toFixed(3) + ')'
                ];
            }}
            
            const option = {{
                title: {{
                    text: 'St. Himark 区域严重程度 - ' + timeData.time_display,
                    left: 'center',
                    top: 20,
                    textStyle: {{
                        fontSize: 18,
                        color: '#2c3e50',
                        fontWeight: 'normal'
                    }}
                }},
                tooltip: {{
                    trigger: 'item',
                    backgroundColor: 'rgba(50, 50, 50, 0.9)',
                    borderColor: '#777',
                    borderWidth: 1,
                    textStyle: {{
                        color: '#fff',
                        fontSize: 14
                    }},
                    formatter: function(params) {{
                        const data = mapData.find(d => d.name === params.name);
                        return data ? data.tooltip : params.name + ': 无数据';
                    }}
                }},
                visualMap: {{
                    min: currentRange.min,
                    max: currentRange.max,
                    left: 'left',
                    bottom: '15%',
                    text: scaleText,
                    textStyle: {{
                        color: '#2c3e50'
                    }},
                    outOfRange: {{
                        color: '#999999'
                    }},
                    calculable: true,
                    realtime: true
                }},
                series: [{{
                    name: '严重程度',
                    type: 'map',
                    map: 'StHimark',
                    data: mapData,
                    roam: false,
                    label: {{
                        show: false
                    }},
                    emphasis: {{
                        label: {{
                            show: true,
                            color: '#fff',
                            fontSize: 12
                        }},
                        itemStyle: {{
                            areaColor: '#ffd54f',
                            borderColor: '#fff',
                            borderWidth: 2
                        }}
                    }},
                    itemStyle: {{
                        borderColor: '#fff',
                        borderWidth: 0.5
                    }}
                }}]
            }};
            
            chart.setOption(option, true);
            currentTimeDisplay.textContent = timeData.time_display;
            timeSlider.value = timeIndex;
            progressInfo.textContent = `第 ${{timeIndex + 1}} 个时间点，共 ${{timeKeys.length}} 个`;
        }}
        
        // 切换数据类型
        function switchDataType(newDataType) {{
            if (currentDataType !== newDataType) {{
                currentDataType = newDataType;
                currentDataTypeDisplay.textContent = dataTypeNames[newDataType];
                
                // 更新按钮状态
                dataTypeBtns.forEach(btn => {{
                    btn.classList.remove('active');
                    if (btn.dataset.type === newDataType) {{
                        btn.classList.add('active');
                    }}
                }});
                
                // 重新渲染地图
                updateMap(currentTimeIndex);
            }}
        }}
        
        // 播放控制
        function play() {{
            if (isPlaying) {{
                clearInterval(playInterval);
                playBtn.textContent = '▶ 播放';
                isPlaying = false;
            }} else {{
                playBtn.textContent = '⏸ 暂停';
                isPlaying = true;
                playInterval = setInterval(() => {{
                    if (currentTimeIndex < timeKeys.length - 1) {{
                        currentTimeIndex++;
                        updateMap(currentTimeIndex);
                    }} else {{
                        // 播放结束
                        clearInterval(playInterval);
                        playBtn.textContent = '▶ 播放';
                        isPlaying = false;
                    }}
                }}, 100);  // 1000ms间隔，1秒一帧
            }}
        }}
        
        // 事件监听
        timeSlider.addEventListener('input', (e) => {{
            if (isPlaying) {{
                clearInterval(playInterval);
                playBtn.textContent = '▶ 播放';
                isPlaying = false;
            }}
            currentTimeIndex = parseInt(e.target.value);
            updateMap(currentTimeIndex);
        }});
        
        playBtn.addEventListener('click', play);
        
        prevBtn.addEventListener('click', () => {{
            if (isPlaying) {{
                clearInterval(playInterval);
                playBtn.textContent = '▶ 播放';
                isPlaying = false;
            }}
            if (currentTimeIndex > 0) {{
                currentTimeIndex--;
                updateMap(currentTimeIndex);
            }}
        }});
        
        nextBtn.addEventListener('click', () => {{
            if (isPlaying) {{
                clearInterval(playInterval);
                playBtn.textContent = '▶ 播放';
                isPlaying = false;
            }}
            if (currentTimeIndex < timeKeys.length - 1) {{
                currentTimeIndex++;
                updateMap(currentTimeIndex);
            }}
        }});
        
        resetBtn.addEventListener('click', () => {{
            if (isPlaying) {{
                clearInterval(playInterval);
                playBtn.textContent = '▶ 播放';
                isPlaying = false;
            }}
            currentTimeIndex = 0;
            updateMap(currentTimeIndex);
        }});
        
        // 数据类型切换事件
        dataTypeBtns.forEach(btn => {{
            btn.addEventListener('click', () => {{
                if (isPlaying) {{
                    clearInterval(playInterval);
                    playBtn.textContent = '▶ 播放';
                    isPlaying = false;
                }}
                switchDataType(btn.dataset.type);
            }});
        }});
        
        // 键盘控制
        document.addEventListener('keydown', (e) => {{
            switch(e.key) {{
                case 'ArrowLeft':
                    if (currentTimeIndex > 0) {{
                        currentTimeIndex--;
                        updateMap(currentTimeIndex);
                    }}
                    break;
                case 'ArrowRight':
                    if (currentTimeIndex < timeKeys.length - 1) {{
                        currentTimeIndex++;
                        updateMap(currentTimeIndex);
                    }}
                    break;
                case ' ':
                    e.preventDefault();
                    play();
                    break;
            }}
        }});
        
        // 初始化地图
        updateMap(0);
        
        // 窗口大小改变时重新调整
        window.addEventListener('resize', () => {{
            chart.resize();
        }});
        
        // 添加加载提示
        console.log('地图已加载完成，包含', timeKeys.length, '个时间点');
        console.log('使用方向键或空格键可以控制时间线');
    </script>
</body>
</html>
"""

# 保存HTML文件
with open("st_himark_map_improved.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("✅ 改进版时间线地图已生成：st_himark_map_improved.html")
print("📊 改进内容：")
print(f"   ✓ 包含全部 {len(unique_times)} 个时间点（每小时）")
print("   ✓ 修复了tooltip换行显示问题（使用<br/>替代\\n）")
print("   ✓ 修复了无数据区域的灰色显示（使用null值）")
print("   ✓ 反转色阶：score越小越严重（红色）")
print("   ✓ 播放速度加快一倍（500ms间隔）")
print("   ✓ 时间线控制器移到地图下方")
print("   ✓ 美化了界面设计，添加渐变和阴影效果")
print("   ✓ 添加了键盘控制（方向键和空格键）")
print("   ✓ 优化了tooltip样式和地图交互效果")
print("   ✓ 添加了数据类型选择器，支持7种数据类型切换")
print("   ✓ 实现了area adjusted列的动态缩放和色阶调整")
print("   ✓ 显示每种数据类型的原始数值范围")
print("")
print("🎛️ 数据类型说明：")
print("   • min_sentiment_score: 使用0-100标准色阶")
for data_key, column_name in data_types.items():
    if column_name in area_adjusted_columns:
        col_range = global_ranges[column_name]
        print(f"   • {data_key}: 动态缩放，原始范围 [{col_range['min']:.3f}, {col_range['max']:.3f}]") 