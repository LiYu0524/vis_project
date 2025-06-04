import json
from shapely.geometry import Polygon
import pandas as pd

# === 1. 读取 Labelme 的 JSON 文件 ===
with open("map.json", "r", encoding="utf-8") as f:
    labelme_data = json.load(f)

area_records = []

# === 2. 遍历每个标注区域，构建 Polygon 并计算面积 ===
for shape in labelme_data["shapes"]:
    name = shape["label"]
    points = shape["points"]

    # 确保 polygon 是闭合的（首尾相连）
    if points[0] != points[-1]:
        points.append(points[0])

    polygon = Polygon(points)
    area = polygon.area

    area_records.append({
        "name": name,
        "area": area
    })

# === 3. 构建 DataFrame 并计算总面积占比 ===
df = pd.DataFrame(area_records)
total_area = df["area"].sum()
df["percentage"] = (df["area"] / total_area) * 100

# === 4. 排序和格式化输出 ===
df = df.sort_values(by="area", ascending=False).reset_index(drop=True)
df["rank"] = df.index + 1
df["area"] = df["area"].round(2)
df["percentage"] = df["percentage"].round(3)
df = df[["rank", "name", "area", "percentage"]]

# === 5. 保存为 CSV 和 JSON（可选） ===
df.to_csv("st_himark_area_from_labelme.csv", index=False)
df.to_json("st_himark_area_from_labelme.json", orient="records", indent=2)

# === 6. 打印前几行预览 ===
print(df.head())
