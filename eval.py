"""
저장된 체크포인트로 테스트 평가만 재실행 — 재학습 없음.

사용법:
    # 개별 실험
    python eval.py --edge A-fb --lookback 36 --features base
    python eval.py --edge A-fb --lookback 36 --features base --addaptadj

    # 전체 실험 일괄 (summary.csv에 있는 모든 행 재평가)
    python eval.py --all
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import argparse
import numpy as np
import torch
import pandas as pd

import util
import graph_utils
from engine import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device',     type=str,  default='cuda:0')
parser.add_argument('--edge',       type=str,  default='A-fb',
                    choices=['identity', 'A-fw', 'A-fb', 'B', 'C', 'D'])
parser.add_argument('--lookback',   type=int,  default=36)
parser.add_argument('--features',   type=str,  default='base',
                    choices=['base', 'elev', 'rf'])
parser.add_argument('--horizon',    type=int,  default=18)
parser.add_argument('--aptonly',    action='store_true')
parser.add_argument('--addaptadj',  action='store_true')
parser.add_argument('--randomadj',  action='store_true')
parser.add_argument('--num_nodes',  type=int,  default=11)
parser.add_argument('--nhid',       type=int,  default=32)
parser.add_argument('--batch_size', type=int,  default=64)
parser.add_argument('--dropout',    type=float, default=0.3)
parser.add_argument('--expid',      type=int,  default=1)
parser.add_argument('--all',        action='store_true',
                    help='summary.csv 의 모든 행을 일괄 재평가')
args = parser.parse_args()


def eval_one(edge, lookback, features, adaptive_mode, device_str='cuda:0'):
    device = torch.device(device_str)

    feat_dim = {"base": 2, "elev": 3, "rf": 3}
    in_dim = feat_dim[features]

    tag = f"edge{edge}_lb{lookback}_feat{features}"
    if adaptive_mode == "aptonly":
        tag += "_aptonly"
    elif adaptive_mode == "adaptive":
        tag += "_adaptive"

    save_prefix = os.path.join("garage", "anseong", tag, "exp1")
    best_path   = f"{save_prefix}_best.pth"

    if not os.path.exists(best_path):
        print(f"  [건너뜀] 체크포인트 없음: {best_path}")
        return None

    supports = graph_utils.get_supports(edge, device=device)
    adjinit  = None if adaptive_mode == "aptonly" else supports[0]
    if adaptive_mode == "aptonly":
        supports = None

    addaptadj = (adaptive_mode in ("adaptive", "aptonly"))

    data_dir   = os.path.join("data", "anseong", f"lb{lookback}_{features}")
    dataloader = util.load_dataset(data_dir, 64, 64, 64)
    scaler     = dataloader['scaler']

    engine = trainer(
        scaler, in_dim, 18, 11,
        32, 0.3, 0.001, 0.0001,
        device, supports, True, addaptadj, adjinit,
    )
    engine.model.load_state_dict(torch.load(best_path, map_location=device))
    engine.model.eval()

    realy = torch.FloatTensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    outputs = []
    for x, _ in dataloader['test_loader'].get_iterator():
        x = torch.FloatTensor(x).to(device).transpose(1, 3)
        with torch.no_grad():
            x_pad = torch.nn.functional.pad(x, (1, 0, 0, 0))
            pred  = engine.model(x_pad).transpose(1, 3)[:, -1:, :, :]
        outputs.append(pred.squeeze(1))
    yhat = torch.cat(outputs, dim=0)[:realy.size(0)]

    amae, armse, amape, anse, ar2 = [], [], [], [], []
    for h in range(18):
        pred = scaler.inverse_transform(yhat[:, :, h])
        real = realy[:, :, h]
        mae, mape, rmse, nse, r2 = util.metric(pred, real)
        amae.append(mae); armse.append(rmse); amape.append(mape)
        anse.append(nse); ar2.append(r2)

    result = {
        "edge": edge, "lookback": lookback, "features": features,
        "adaptive": adaptive_mode,
        "test_avg_mae":  round(float(np.mean(amae)),  4),
        "test_avg_rmse": round(float(np.mean(armse)), 4),
        "test_avg_mape": round(float(np.mean(amape)), 4),
        "test_avg_nse":  round(float(np.mean(anse)),  4),
        "test_avg_r2":   round(float(np.mean(ar2)),   4),
    }
    print(f"  [{tag}]  MAE={result['test_avg_mae']}  RMSE={result['test_avg_rmse']}"
          f"  NSE={result['test_avg_nse']}  R2={result['test_avg_r2']}")
    return result


def update_csv(results: list[dict]):
    csv_path = os.path.join("results", "summary.csv")
    if not os.path.exists(csv_path):
        print("  summary.csv 없음 — 건너뜀")
        return

    df = pd.read_csv(csv_path)
    for col in ["test_avg_nse", "test_avg_r2"]:
        if col not in df.columns:
            df[col] = float("nan")

    for r in results:
        if r is None:
            continue
        mask = (
            (df["edge"]     == r["edge"]) &
            (df["lookback"] == r["lookback"]) &
            (df["features"] == r["features"]) &
            (df["adaptive"] == r["adaptive"])
        )
        if mask.sum() == 0:
            print(f"  [경고] CSV에 해당 행 없음: {r}")
            continue
        df.loc[mask, "test_avg_nse"] = r["test_avg_nse"]
        df.loc[mask, "test_avg_r2"]  = r["test_avg_r2"]

    # 컬럼 순서 고정
    cols = ["edge", "lookback", "features", "adaptive",
            "val_mae", "test_avg_mae", "test_avg_rmse",
            "test_avg_mape", "test_avg_nse", "test_avg_r2",
            "best_epoch", "n_params"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(csv_path, index=False)
    print(f"  summary.csv 업데이트 완료: {csv_path}")


def main():
    if args.all:
        csv_path = os.path.join("results", "summary.csv")
        if not os.path.exists(csv_path):
            print("summary.csv 없음. 먼저 실험을 실행하세요.")
            return
        df = pd.read_csv(csv_path)
        results = []
        for _, row in df.iterrows():
            print(f"\n평가: edge={row['edge']}  lb={row['lookback']}"
                  f"  feat={row['features']}  adaptive={row['adaptive']}")
            r = eval_one(row["edge"], int(row["lookback"]),
                         row["features"], row["adaptive"], args.device)
            results.append(r)
        update_csv(results)
    else:
        adaptive_mode = ("aptonly"  if args.aptonly  else
                         "adaptive" if args.addaptadj else "none")
        r = eval_one(args.edge, args.lookback, args.features,
                     adaptive_mode, args.device)
        if r:
            update_csv([r])


if __name__ == "__main__":
    main()
