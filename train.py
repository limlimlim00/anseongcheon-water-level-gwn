"""
안성천 수위 예측 — Graph WaveNet 학습 스크립트

사용법:
    # Edge A-fb (수계 위상 directed, forward+backward)
    python train.py --edge A-fb --lookback 72

    # Adaptive only (사전 정보 없이 학습)
    python train.py --edge A-fb --lookback 72 --aptonly --addaptadj --randomadj

    # Best + Adaptive
    python train.py --edge A-fb --lookback 72 --addaptadj

엣지 종류:
    identity  연결 없음 (공간정보 제거)
    A-fw      수계 위상 directed, forward-only
    A-fb      수계 위상 directed, forward+backward
    B         수계 위상 undirected
    C         지리적 거리 Gaussian undirected
    D         Pearson 상관 undirected
"""

import sys
sys.stdout.reconfigure(line_buffering=True)   # pipe에서도 즉시 출력

import os
import time
import argparse
import numpy as np
import torch

import util
import graph_utils
from engine import trainer

# ── 인자 ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--device',     type=str,   default='cuda:0')
parser.add_argument('--edge',       type=str,   default='A-fb',
                    choices=['identity', 'A-fw', 'A-fb', 'B', 'C', 'D'],
                    help='엣지 설계 유형')
parser.add_argument('--lookback',   type=int,   default=72,
                    help='입력 시퀀스 길이 (36 | 72 | 144)')
parser.add_argument('--features',   type=str,   default='base',
                    choices=['base', 'elev', 'rf'],
                    help='노드 피처 구성 (base=수위+tod, elev=+해발고도, rf=+강수량최근접)')
parser.add_argument('--horizon',    type=int,   default=18,
                    help='예측 horizon (고정: 18)')
parser.add_argument('--gcn_bool',   action='store_true', default=True,
                    help='GCN 레이어 사용')
parser.add_argument('--aptonly',    action='store_true',
                    help='Adaptive adj 만 사용 (predefined 제거)')
parser.add_argument('--addaptadj',  action='store_true',
                    help='Adaptive adj 추가 학습')
parser.add_argument('--randomadj',  action='store_true',
                    help='Adaptive adj 랜덤 초기화 (False면 predefined로 초기화)')
parser.add_argument('--num_nodes',  type=int,   default=11)
parser.add_argument('--in_dim',     type=int,   default=None,
                    help='입력 피처 수 (미지정 시 features에서 자동 결정)')
parser.add_argument('--nhid',       type=int,   default=32)
parser.add_argument('--batch_size', type=int,   default=64)
parser.add_argument('--lr',         type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--dropout',    type=float, default=0.3)
parser.add_argument('--epochs',     type=int,   default=100)
parser.add_argument('--patience',   type=int,   default=15,
                    help='Early stopping patience (val loss 기준)')
parser.add_argument('--print_every', type=int,  default=100)
parser.add_argument('--seed',       type=int,   default=42)
parser.add_argument('--expid',      type=int,   default=1)
args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # ── in_dim 자동 결정 ─────────────────────────────────────
    feat_dim = {"base": 2, "elev": 3, "rf": 3}
    in_dim = args.in_dim if args.in_dim is not None else feat_dim[args.features]

    # ── 저장 경로 ────────────────────────────────────────────
    tag = f"edge{args.edge}_lb{args.lookback}_feat{args.features}"
    if args.aptonly:
        tag += "_aptonly"
    elif args.addaptadj:
        tag += "_adaptive"
    save_dir = os.path.join("garage", "anseong", tag)
    os.makedirs(save_dir, exist_ok=True)
    save_prefix = os.path.join(save_dir, f"exp{args.expid}")

    # ── 인접행렬 (support) ───────────────────────────────────
    supports = graph_utils.get_supports(args.edge, device=device)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    # ── 데이터 ───────────────────────────────────────────────
    data_dir = os.path.join("data", "anseong", f"lb{args.lookback}_{args.features}")
    dataloader = util.load_dataset(data_dir, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    # ── 모델 ─────────────────────────────────────────────────
    engine = trainer(
        scaler, in_dim, args.horizon, args.num_nodes,
        args.nhid, args.dropout, args.lr, args.weight_decay,
        device, supports, args.gcn_bool, args.addaptadj, adjinit,
    )

    total_params = sum(p.numel() for p in engine.model.parameters() if p.requires_grad)
    print(f"[{tag}]  device={device}  params={total_params:,}")
    print(f"  data={data_dir}  train={len(dataloader['x_train']):,}  "
          f"val={len(dataloader['x_val']):,}  test={len(dataloader['x_test']):,}")

    # ── 학습 ─────────────────────────────────────────────────
    his_loss = []
    no_improve = 0
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        dataloader['train_loader'].shuffle()

        train_loss, train_mape, train_rmse = [], [], []
        train_batches = list(dataloader['train_loader'].get_iterator())
        n_batches = len(train_batches)
        for i, (x, y) in enumerate(train_batches):
            x = torch.FloatTensor(x).to(device).transpose(1, 3)   # (B, F, N, T)
            y = torch.FloatTensor(y).to(device).transpose(1, 3)   # (B, 1, N, horizon)
            loss, mape, rmse = engine.train(x, y[:, 0, :, :])
            train_loss.append(loss)
            train_mape.append(mape)
            train_rmse.append(rmse)
            is_last = (i == n_batches - 1)
            if i % args.print_every == 0 or is_last:
                print(f"  Iter {i:04d}/{n_batches-1}  loss={loss:.4f}  mape={mape:.4f}  rmse={rmse:.4f}", flush=True)

        val_loss, val_mape, val_rmse = [], [], []
        for x, y in dataloader['val_loader'].get_iterator():
            x = torch.FloatTensor(x).to(device).transpose(1, 3)
            y = torch.FloatTensor(y).to(device).transpose(1, 3)
            loss, mape, rmse = engine.eval(x, y[:, 0, :, :])
            val_loss.append(loss)
            val_mape.append(mape)
            val_rmse.append(rmse)

        t2 = time.time()
        mv_loss = np.mean(val_loss)
        his_loss.append(mv_loss)

        print(f"Epoch {epoch:03d}  "
              f"train_loss={np.mean(train_loss):.4f}  "
              f"val_loss={mv_loss:.4f}  val_mape={np.mean(val_mape):.4f}  "
              f"val_rmse={np.mean(val_rmse):.4f}  "
              f"time={t2-t0:.1f}s", flush=True)

        torch.save(engine.model.state_dict(),
                   f"{save_prefix}_ep{epoch}_{mv_loss:.4f}.pth")

        # Early stopping
        if mv_loss < best_val:
            best_val = mv_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # ── 테스트 ───────────────────────────────────────────────
    best_epoch = int(np.argmin(his_loss))
    best_file  = f"{save_prefix}_ep{best_epoch+1}_{his_loss[best_epoch]:.4f}.pth"
    engine.model.load_state_dict(torch.load(best_file))
    print(f"\n▶ Best epoch={best_epoch+1}  val_loss={his_loss[best_epoch]:.4f}")

    realy = torch.FloatTensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]   # (N_test, N, horizon)

    outputs = []
    for x, y in dataloader['test_loader'].get_iterator():
        x = torch.FloatTensor(x).to(device).transpose(1, 3)
        with torch.no_grad():
            x_pad = torch.nn.functional.pad(x, (1, 0, 0, 0))
            pred = engine.model(x_pad).transpose(1, 3)[:, -1:, :, :]  # (B, 1, N, horizon)
        outputs.append(pred.squeeze(1))               # (B, N, horizon)
    yhat = torch.cat(outputs, dim=0)[:realy.size(0)]  # (N_test, N, horizon)

    amae, armse, amape, anse, ar2 = [], [], [], [], []
    for h in range(args.horizon):
        pred = scaler.inverse_transform(yhat[:, :, h])
        real = realy[:, :, h]
        mae, mape, rmse, nse, r2 = util.metric(pred, real)
        amae.append(mae)
        armse.append(rmse)
        amape.append(mape)
        anse.append(nse)
        ar2.append(r2)
        print(f"  horizon {h+1:02d}  MAE={mae:.4f}  RMSE={rmse:.4f}  NSE={nse:.4f}  R2={r2:.4f}")

    print(f"\n  avg MAE={np.mean(amae):.4f}  avg RMSE={np.mean(armse):.4f}"
          f"  avg NSE={np.mean(anse):.4f}  avg R2={np.mean(ar2):.4f}")

    final_path = f"{save_prefix}_best.pth"
    torch.save(engine.model.state_dict(), final_path)
    print(f"  저장: {final_path}")

    # ── 결과 CSV 기록 ─────────────────────────────────────────
    import csv, pandas as pd
    csv_path = os.path.join("results", "summary.csv")
    os.makedirs("results", exist_ok=True)

    fieldnames = ["edge", "lookback", "features", "adaptive",
                  "val_mae", "test_avg_mae", "test_avg_rmse",
                  "test_avg_mape", "test_avg_nse", "test_avg_r2",
                  "best_epoch", "n_params"]

    # 기존 CSV에 nse/r2 컬럼 없으면 마이그레이션
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        for col in ["test_avg_nse", "test_avg_r2"]:
            if col not in existing.columns:
                existing[col] = ""
        existing = existing[fieldnames]
        existing.to_csv(csv_path, index=False)

    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()

        adaptive_mode = "aptonly" if args.aptonly else ("adaptive" if args.addaptadj else "none")

        writer.writerow({
            "edge":          args.edge,
            "lookback":      args.lookback,
            "features":      args.features,
            "adaptive":      adaptive_mode,
            "val_mae":       round(his_loss[best_epoch], 6),
            "test_avg_mae":  round(float(np.mean(amae)),  4),
            "test_avg_rmse": round(float(np.mean(armse)), 4),
            "test_avg_mape": round(float(np.mean(amape)), 4),
            "test_avg_nse":  round(float(np.mean(anse)),  4),
            "test_avg_r2":   round(float(np.mean(ar2)),   4),
            "best_epoch":    best_epoch + 1,
            "n_params":      total_params,
        })
    print(f"  결과 기록: {csv_path}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal: {time.time()-t0:.1f}s")
