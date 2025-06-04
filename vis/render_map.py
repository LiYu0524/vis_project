import json
from pyecharts.charts import Map
from pyecharts import options as opts

# 读取 GeoJSON 文件
with open("st_himark.geojson", "r") as f:
    geojson_data = json.load(f)

# 假设你填入的严重程度数据如下（可以修改）
severity_data = {
    "Old Town": 85,
    "Downtown": 60,
    "Palace Hills": 30,
    "Northwest": 50,
    "Weston": 40,
    "Southwest": 70,
    "Safe Town": 90,
    "Easton": 45,
    "East Parton": 35,
    "West Parton": 65,
    "Oak Willow": 55,
    "Southton": 40,
    "Scenic Vista": 75,
    "Broadview": 50,
    "Chapparal": 60,
    "Cheddarford": 55,
    "Pepper Mill": 45,
    "Terrapin Springs": 50,
    "Wilson Forest": 20
}

# 转为 pyecharts 需要的数据格式
data = list(severity_data.items())

# 创建地图对象
map_chart = (
    Map(init_opts=opts.InitOpts(width="1200px", height="900px"))
    .add_js_funcs("echarts.registerMap('StHimark', {});".format(json.dumps(geojson_data)))
    .add("Severity", data, maptype="StHimark")
    .set_global_opts(
        title_opts=opts.TitleOpts(title="St. Himark 区域严重程度"),
        visualmap_opts=opts.VisualMapOpts(
            min_=0,
            max_=100,
            is_piecewise=False,
            pos_left="left",
            pos_top="bottom",
        )
    )
)

# 保存为 HTML 网页
map_chart.render("st_himark_map.html")
print("✅ 地图已生成：st_himark_map.html")
