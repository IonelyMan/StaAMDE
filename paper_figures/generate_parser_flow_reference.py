"""Draw an editable SVG and a PNG reference for the paper's APK input flow."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.font_manager import FontProperties


OUT = Path(__file__).resolve().parent
FONT = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
BOLD = FontProperties(fname=r"C:\Windows\Fonts\msyhbd.ttc")

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["figure.facecolor"] = "white"

fig, ax = plt.subplots(figsize=(17, 7), dpi=100)
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(0, 1700)
ax.set_ylim(700, 0)
ax.axis("off")


def box(x, y, w, h, face, edge, radius=18, lw=2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=lw, edgecolor=edge, facecolor=face,
        )
    )


def txt(x, y, value, size=20, color="#213445", bold=False, align="left"):
    ax.text(
        x, y, value, fontsize=size, color=color,
        fontproperties=BOLD if bold else FONT,
        ha=align, va="center",
    )


def arrow(x1, y1, x2, y2, color="#526C7C"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=20,
            linewidth=2.4, color=color, shrinkA=0, shrinkB=0,
        )
    )


txt(58, 49, "APK 静态解析与两个分支的输入", 28, bold=True)
ax.plot([58, 1640], [75, 75], color="#D8E3E9", lw=2)

# The APK contents are deliberately INSIDE the APK card. They are inputs to
# Androguard, never drawn as files exported by it.
box(58, 168, 292, 340, "#F7FBFD", "#8EADBD")
txt(84, 207, "① 原始 APK", 23, "#235D78", True)
txt(84, 255, "安装包内已有：", 18)
box(84, 283, 240, 54, "#E5F1F7", "#BCD4E0", 10, 1.4)
txt(204, 310, "AndroidManifest.xml", 17, "#235D78", align="center")
box(84, 355, 240, 54, "#E5F1F7", "#BCD4E0", 10, 1.4)
txt(204, 382, "classes*.dex", 19, "#235D78", align="center")
txt(84, 454, "XML 和 DEX 属于 APK", 17, "#547080")

arrow(351, 338, 430, 338)

box(430, 188, 300, 300, "#F9F6FC", "#A99AC0")
txt(455, 227, "② Androguard", 23, "#654E7D", True)
txt(455, 266, "AnalyzeAPK(apk_path)", 17, "#654E7D")
ax.plot([455, 705], [289, 289], color="#DDD3E7", lw=1.5)
txt(455, 328, "读取并解码 Manifest", 18)
txt(455, 372, "解析 DEX 字节码", 18)
txt(455, 416, "建立方法等交叉引用", 18)
txt(455, 461, "产生内存中的解析对象", 16, "#716482")

arrow(731, 338, 812, 338)

box(812, 148, 370, 380, "#F5FBF8", "#88B5A1")
txt(838, 189, "③ 解析后可用的信息", 22, "#317257", True)
box(837, 220, 320, 104, "#E7F4EC", "#C0DBCA", 10, 1.2)
txt(855, 246, "清单信息", 18, "#317257", True)
txt(855, 287, "权限 / 组件 / Intent", 18)
box(837, 340, 320, 141, "#E7F4EC", "#C0DBCA", 10, 1.2)
txt(855, 368, "DEX 分析信息", 18, "#317257", True)
txt(855, 409, "方法调用 / 操作码", 18)
txt(855, 446, "字符串等", 18)

# One parsed information set feeds both branches.
ax.plot([1183, 1222], [338, 338], color="#526C7C", lw=2.4)
ax.plot([1222, 1222], [235, 505], color="#526C7C", lw=2.4)
arrow(1222, 235, 1260, 235)
arrow(1222, 505, 1260, 505)

box(1260, 114, 380, 240, "#FFF9F3", "#DBAE81")
txt(1285, 151, "④ 决策树分支", 22, "#9E642C", True)
txt(1285, 194, "静态特征报告（JSON）", 18)
txt(1285, 232, "特征筛选与编码", 18)
box(1284, 264, 330, 61, "#FCEBD9", "#EDD0B1", 10, 1)
txt(1449, 294, "输入：特征向量 x_static", 17, "#874E1C", True, "center")

box(1260, 385, 380, 240, "#F5F8FE", "#91ACD4")
txt(1285, 422, "④ 图模型分支", 22, "#446897", True)
txt(1285, 465, "异构图构建（PT）", 18)
txt(1285, 503, "节点、边与关系整理", 18)
box(1284, 535, 330, 61, "#E5EDFB", "#BECEEA", 10, 1)
txt(1449, 565, "输入：APK 异构图 G_APK", 17, "#365884", True, "center")

txt(
    58, 665,
    "说明：两条构建程序分别调用 AnalyzeAPK；均从 APK 读取内容并解析，不单独导出 XML 或 DEX 文件。",
    16, "#516879",
)

for suffix in ("png", "svg"):
    fig.savefig(OUT / f"clear_parser_flow_reference.{suffix}", dpi=100, facecolor="white")
plt.close(fig)
