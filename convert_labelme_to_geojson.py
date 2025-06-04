import json
import geojson
from shapely.geometry import Polygon

# 读取 Labelme 的 JSON 文件
with open("map.json", "r") as f:
    labelme_data = json.load(f)

features = []

# 遍历每个标注区域
for shape in labelme_data["shapes"]:
    label = shape["label"]
    points = shape["points"]

    # GeoJSON 格式要求闭合 polygon：首尾点一致
    if points[0] != points[-1]:
        points.append(points[0])

    polygon = Polygon(points)
    feature = geojson.Feature(geometry=polygon, properties={"name": label})
    features.append(feature)

# 构造 FeatureCollection 并保存
geojson_data = geojson.FeatureCollection(features)

with open("st_himark.geojson", "w") as f:
    geojson.dump(geojson_data, f)

print("✅ GeoJSON 已保存为 st_himark.geojson")
