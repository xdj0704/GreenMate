"""趋势模型：对等间隔时序做最小二乘线性拟合，输出斜率与相关系数。"""


def linear_regression(xs, ys):
    """最小二乘直线拟合，返回 (slope, intercept, r)。"""
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0), 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0:
        return 0.0, my, 0.0
    slope = sxy / sxx
    intercept = my - slope * mx
    r = sxy / (sxx * syy) ** 0.5 if syy > 0 else 0.0
    return slope, intercept, r


def trend_stats(series):
    """series: list[float]，等间隔（每小时）取值。返回趋势摘要。"""
    if not series:
        return {}
    xs = list(range(len(series)))
    slope, _intercept, r = linear_regression(xs, series)
    return {
        "start": series[0],
        "end": series[-1],
        "delta": series[-1] - series[0],
        "rate_per_hour": slope,
        "r": r,
        "monotonic": all(b >= a for a, b in zip(series, series[1:])),
    }