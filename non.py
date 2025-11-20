# xianle_products_chart.py
# 仙乐健康 2019–2023 各产品线收入曲线图（解决中文乱码，适配 macOS）

import os
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ===================== 1. 选择 / 检查中文字体 =====================

# 按顺序尝试的字体路径（Mac 常见位置）
CANDIDATE_FONT_PATHS = [
    "/System/Library/Fonts/STHeiti Medium.ttc",     # 华文黑体/黑体
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/PingFang.ttc",           # 苹方
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Microsoft YaHei.ttf",           # 如果你自己装了微软雅黑
]

font_path = None
for p in CANDIDATE_FONT_PATHS:
    if os.path.exists(p):
        font_path = p
        break

if font_path is None:
    raise FileNotFoundError(
        "没有在常见路径找到中文字体文件，请手动修改脚本中的 CANDIDATE_FONT_PATHS，"
        "把你 Mac 上实际的字体路径（例如 /System/Library/Fonts/STHeiti Medium.ttc）填进去。"
    )

# 用这个字体做全局中文字体
zh_font = fm.FontProperties(fname=font_path)

# 解决负号显示问题
plt.rcParams["axes.unicode_minus"] = False

# ===================== 2. 数据 =====================

years = [2019, 2020, 2021, 2022, 2023]

soft_capsule = [8.23, 8.83, 9.03, 9.91, 15.27]  # 软胶囊（亿元）
tablet       = [2.66, 2.91, 3.20, 3.41, 3.62]  # 片剂
powder       = [2.41, 3.12, 3.29, 1.92, 2.93]  # 粉剂
gummy        = [1.17, 3.27, 4.72, 5.93, 7.40]  # 营养软糖
drink        = [0.48, 1.83, 2.50, 3.38, 4.22]  # 功能饮品

# ===================== 3. 画图 =====================

plt.figure(figsize=(8, 5))

# 这里颜色只是为了方便区分，顺序固定：
# 蓝：软胶囊，绿：片剂，橙：粉剂，红：营养软糖，紫：功能饮品
plt.plot(years, soft_capsule, marker="o", label="软胶囊", color="blue")
plt.plot(years, tablet,       marker="o", label="片剂",   color="green")
plt.plot(years, powder,       marker="o", label="粉剂",   color="orange")
plt.plot(years, gummy,        marker="o", label="营养软糖", color="red")
plt.plot(years, drink,        marker="o", label="功能饮品", color="purple")

plt.title("仙乐健康 2019–2023 年主要产品线销售收入（亿元）",
          fontproperties=zh_font, fontsize=14)
plt.xlabel("年份", fontproperties=zh_font, fontsize=12)
plt.ylabel("销售收入（亿元）", fontproperties=zh_font, fontsize=12)

# 坐标轴刻度也用中文字体
plt.xticks(years, [str(y) for y in years], fontproperties=zh_font)
plt.yticks(fontproperties=zh_font)

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(prop=zh_font, fontsize=10)

plt.tight_layout()

# ===================== 4. 保存图片 =====================

out_file = "xianle_product_revenue_2019_2023.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print(f"已生成图片文件：{out_file}")
