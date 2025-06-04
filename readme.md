# Vis_Poj(Chinese ver English ver seen below)

ST_Himark 地震可视化项目

## vis 目录说明

使用 `labelme` 工具对地图进行标注，生成 JSON 格式的标注文件。

`convert_labelme_to_geojson.py` 用于将 JSON 标注文件转换为 pyecharts 可读取的 GeoJSON 格式。

### 地图渲染

`render_map_improved.py` 通过 pyecharts 渲染地图。

需要将 `min_sentiment_score_for_all_locations_area_wgt.csv` 文件放在同一目录下。

`st_himark_map_improved.html` 是最终生成的可视化结果页面。


### 辅助功能

`render_map.py` 用于测试地图渲染功能。

`calculate_area.py` 用于计算各区域的面积占比。

# Vis_Poj (English ver)

Visualization for ST_himark Earthquake

## vis

`labelme` is used to create map annotation in JSON format

`convert_labelme_to_geojson.py`  convert the json format to format which could read by pyechart.

### render map

`render_map_improved.py` rendering the map through pyechart

`min_sentiment_score_for_all_locations_area_wgt.csv` needs to be placed in the same directory.

`st_himark_map_improved.html` is the final result for visualization.

### support function

`render_map.py` used for testing

`calculate_area.py` is used to calculate the area percentage of each region.

