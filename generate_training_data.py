"""
안성천 수위 데이터 → GWN 학습용 슬라이딩 윈도우 .npz 생성

사용법:
    python generate_training_data.py --lookback 72 --features base
    python generate_training_data.py --lookback 72 --features elev
    python generate_training_data.py --lookback 72 --features rf

피처 구성:
    base : [수위, time_of_day]              F=2
    elev : [수위, time_of_day, 해발고도]    F=3
    rf   : [수위, time_of_day, 강수량]      F=3  (최근접 관측소 매핑)

출력:
    data/anseong/lb{lookback}_{features}/train.npz
    data/anseong/lb{lookback}_{features}/val.npz
    data/anseong/lb{lookback}_{features}/test.npz

각 .npz 구조:
    x : (samples, lookback, N, F)   채널 0=수위(raw), 1=tod, 2=해발고도or강수량
    y : (samples, horizon, N, 1)    수위 원시값

분할 기준 (y의 마지막 타임스텝 기준):
    Train  2019-01-01 ~ 2022-12-31
    Val    2023-01-01 ~ 2023-12-31
    Test   2024-01-01 ~ 2024-12-31
"""

import argparse
import os
import numpy as np
import pandas as pd

WL_PATH          = "api/data/waterlevel_10m_filled.csv"
RF_PATH          = "api/data/rainfall_mapped_10m.csv"
HORIZON  = 18

TRAIN_END = "2022-12-31 23:50"
VAL_END   = "2023-12-31 23:50"
TEST_END  = "2024-12-31 23:50"

# 해발고도 (El.m), 관측소 순서 = waterlevel_10m_filled.csv 컬럼 순서
ELEVATION = np.array([13.85, 18.35, 8.81, 3.96, 2.02,
                      11.61,  5.24, 6.37, 5.34, 1.96, 1.98], dtype=np.float32)


def build_windows(wl: np.ndarray, tod: np.ndarray, extra: np.ndarray | None,
                  lookback: int, horizon: int, index: pd.DatetimeIndex):
    """
    슬라이딩 윈도우 생성.

    Args:
        wl    : (T, N) 수위 원시값
        tod   : (T, N) time_of_day
        extra : (T, N) 추가 피처 (해발고도 or 강수량), None이면 base
        lookback, horizon: 윈도우 크기

    Returns:
        x      : (S, lookback, N, F)
        y      : (S, horizon,  N, 1)
        t_last : (S,) datetime64  y 마지막 타임스텝
    """
    T = wl.shape[0]
    channels = [wl[:, :, None], tod[:, :, None]]
    if extra is not None:
        channels.append(extra[:, :, None])
    feat = np.concatenate(channels, axis=-1).astype(np.float32)  # (T, N, F)

    x_list, y_list, t_last = [], [], []
    for t in range(lookback - 1, T - horizon):
        x_list.append(feat[t - lookback + 1 : t + 1])
        y_list.append(wl[t + 1 : t + 1 + horizon, :, None])
        t_last.append(index[t + horizon])

    x = np.stack(x_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.float32)
    t_last = np.array(t_last, dtype="datetime64[ns]")
    return x, y, t_last


def generate(args):
    lookback = args.lookback
    features = args.features
    out_dir  = os.path.join("data", "anseong", f"lb{lookback}_{features}")
    os.makedirs(out_dir, exist_ok=True)

    # ── 수위 로드 ────────────────────────────────────────────
    df = pd.read_csv(WL_PATH, index_col=0, parse_dates=True)
    df = df[df.index <= TEST_END]
    wl = df.values.astype(np.float32)   # (T, N)
    T, N = wl.shape

    # time_of_day
    tod_raw = (
        (df.index.values - df.index.values.astype("datetime64[D]"))
        / np.timedelta64(1, "D")
    ).astype(np.float32)
    tod = np.tile(tod_raw[:, None], (1, N))  # (T, N)

    # ── 추가 피처 ────────────────────────────────────────────
    extra = None

    if features == "elev":
        # 해발고도: 정규화 후 (T, N)으로 broadcast
        elev_norm = (ELEVATION - ELEVATION.min()) / (ELEVATION.max() - ELEVATION.min())
        extra = np.tile(elev_norm[None, :], (T, 1)).astype(np.float32)

    elif features == "rf":
        # 강수량: 훈련 구간 max 정규화 → [0, 1]
        # 강수량은 98.5% 가 0mm인 극단적 우편향 분포 → z-score 부적합 (z_max ≈ 97)
        rf_src = RF_PATH
        rf_df = pd.read_csv(rf_src, index_col=0, parse_dates=True)
        rf_df = rf_df[rf_df.index <= TEST_END][list(df.columns)]
        rf = rf_df.values.astype(np.float32)

        train_mask = rf_df.index <= TRAIN_END
        rf_max = rf[train_mask].max()
        rf_max = rf_max if rf_max > 0 else 1.0
        extra  = rf / rf_max   # [0, 1], 대부분 0에 가까움

    # ── 윈도우 생성 ──────────────────────────────────────────
    print(f"데이터: {df.shape}  피처={features}  lookback={lookback}")
    x, y, t_last = build_windows(wl, tod, extra, lookback, HORIZON, df.index)
    print(f"전체 윈도우: x={x.shape}  y={y.shape}  F={x.shape[-1]}")

    splits = {
        "train": t_last <= np.datetime64(TRAIN_END),
        "val"  : (t_last > np.datetime64(TRAIN_END)) & (t_last <= np.datetime64(VAL_END)),
        "test" : (t_last > np.datetime64(VAL_END))   & (t_last <= np.datetime64(TEST_END)),
    }
    for cat, mask in splits.items():
        path = os.path.join(out_dir, f"{cat}.npz")
        np.savez_compressed(path, x=x[mask], y=y[mask])
        print(f"  {cat:5s}  x={x[mask].shape}  → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, default=72)
    parser.add_argument("--features", type=str, default="base",
                        choices=["base", "elev", "rf"],
                        help="base=수위+tod  elev=+해발고도  rf=+강수량(최근접)")
    args = parser.parse_args()
    generate(args)
