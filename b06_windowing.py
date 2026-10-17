"""GreenMate B06 · 窗口化处理脚本（开发步骤 1）。

基于 B02 环境时序样本，按时间戳排序，固定窗口大小：输入 6 个点（3 小时），
预测未来 2~4 个点（1~2 小时）。支持修改 window_size / horizon 重新生成样本。

用法（用 Python 3.12 运行）：
    py -3.12 b06_windowing.py                 # 默认 window_size=6, horizon=2
    py -3.12 b06_windowing.py --window 6 --horizon 4 --out b06_windows.csv
"""

from __future__ import annotations

import argparse
import sys

import b06_ml as ml


def main() -> int:
    ap = argparse.ArgumentParser(description="GreenMate B06 窗口化")
    ap.add_argument("--window", type=int, default=ml.WINDOW_SIZE, help="输入窗口点数，默认 6（3h）")
    ap.add_argument("--horizon", type=int, default=ml.FORECAST_HORIZON, help="预测未来点数，2~4（1~2h）")
    ap.add_argument("--out", type=str, default=ml.DATASET_PATH, help="输出 CSV 路径")
    args = ap.parse_args()

    print("=" * 72)
    print("GreenMate B06 · 步骤 1 窗口化")
    print("=" * 72)

    samples = ml.load_samples()
    print(f"原始时序样本：{len(samples)} 条（{samples['timestamp'].iloc[0]} ~ "
          f"{samples['timestamp'].iloc[-1]}）")

    df = ml.build_windows(samples, window_size=args.window, horizon=args.horizon)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"生成窗口样本：{len(df)} 条 → 已保存 {args.out}")
    print(f"标签分布(NORMAL/WATCH)：{df['label'].value_counts().to_dict()}")
    print(f"特征列（{len(ml.FEATURE_COLS)} 个，仅来自输入窗口）："
          f"{[c.split('_')[0] + '_' + c.split('_')[-1] for c in ml.FEATURE_COLS[:6]]} ...")
    print("\n核心特征说明（可解释的 3 个）：")
    print("  - temperature_current：输入窗口最后一刻气温，直接反映当前热状态")
    print("  - humidity_current：输入窗口最后一刻湿度，涌青/干燥即时表征")
    print("  - temperature_delta：窗口首末气温变化，捕捉升温/降温趋势")
    return 0


if __name__ == "__main__":
    sys.exit(main())