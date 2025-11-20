import matplotlib.pyplot as plt

def plot_revenue_trend(years, revenue_rmb, save_path=None):
    """
    Plot revenue trend line (simple version, integer years).
    :param years: list[int] – e.g. [2019, 2020, 2021, 2022, 2023, 2024]
    :param revenue_rmb: list[float] – revenue in RMB yuan
    :param save_path: optional path to save file
    """
    revenue_100m = [x / 1e8 for x in revenue_rmb]  # 转换为亿元
    plt.figure(figsize=(8, 5))
    plt.plot(years, revenue_100m, marker="o", linewidth=2, color="#1f77b4")
    plt.title("Baihe Bio Revenue Trend (2019–2024)", fontsize=13)
    plt.xlabel("Year", fontsize=11)
    plt.ylabel("Revenue (RMB 100 million)", fontsize=11)
    plt.xticks(years)  # 只显示整数年份
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    # 百合股份营业收入数据（单位：元）
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    revenue_rmb = [
        724_000_000,
        812_000_000,
        655_000_000,
        719_000_000,
        871_000_000,
        801_000_000,
    ]

    plot_revenue_trend(years, revenue_rmb)
    # 若要保存为图片，可用：
    plot_revenue_trend(years, revenue_rmb, save_path="baihe_revenue_trend.png")
