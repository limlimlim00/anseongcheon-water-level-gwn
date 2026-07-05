#!/usr/bin/env bash
# =============================================================
# 안성천 수위 예측 — 전체 실험 자동 실행
#
# 사용법:
#   bash run_experiments.sh           # Phase 0~4 처음부터 끝까지
#   bash run_experiments.sh phase0    # Window size 그리드서치
#   bash run_experiments.sh phase1    # 엣지 설계 비교 (E0~E5)
#   bash run_experiments.sh phase2    # Adaptive 비교 (base / apt / apt+base)
#   bash run_experiments.sh phase3    # 노드 피처: 해발고도
#   bash run_experiments.sh phase4    # 강수량 피처
#
# 실행 순서:
#   Phase 0 → best_lb 자동 선택
#   Phase 1 → best_edge 자동 선택
#   Phase 2, 3, 4 (best_lb, best_edge 고정)
# =============================================================

set -e
cd "$(dirname "$0")"

PYTHON="conda run --no-capture-output -n water-level python"

# ── 학습 하이퍼파라미터 ────────────────────────────────────────
DEVICE="cuda:0"
EPOCHS=100
PATIENCE=15
BATCH=64
SEED=42

# =============================================================
# 유틸
# =============================================================
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

RUN_TS=$(date '+%Y%m%d_%H%M%S')
MASTER_LOG="${LOG_DIR}/run_${RUN_TS}.log"

log() {
    local msg="[$(date '+%H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$MASTER_LOG"
}
sep() {
    local line="$(echo; printf '━%.0s' {1..45})"
    echo "$line"
    echo "$line" >> "$MASTER_LOG"
}

# 개별 실험 실행: 공통 하이퍼파라미터 자동 주입, 로그 이중 저장
run() {
    local edge="" lb="" feat=""
    local args=("$@")
    for ((i=0; i<${#args[@]}; i++)); do
        case "${args[i]}" in
            --edge)     edge="${args[i+1]}" ;;
            --lookback) lb="${args[i+1]}" ;;
            --features) feat="${args[i+1]}" ;;
        esac
    done
    local tag="${edge}_lb${lb}_${feat}"
    local exp_log="${LOG_DIR}/${tag}_$(date '+%Y%m%d_%H%M%S').log"

    log "▶ train.py $* → $exp_log"
    PYTHONUNBUFFERED=1 $PYTHON train.py "$@" \
        --device      "$DEVICE"   \
        --epochs      "$EPOCHS"   \
        --patience    "$PATIENCE" \
        --batch_size  "$BATCH"    \
        --print_every 500         \
        --seed        "$SEED" 2>&1 | tee -a "$exp_log" | tee -a "$MASTER_LOG"
    log "✓ 완료: $exp_log"
}

# results/summary.csv에서 val_mae 기준 최솟값의 특정 컬럼 반환
# 인자: col  pandas_query_string
best_from_csv() {
    local col="$1"
    local cond="$2"
    $PYTHON - <<EOF
import pandas as pd, sys
try:
    df = pd.read_csv("results/summary.csv")
    sub = df.query("$cond")
    if sub.empty:
        print("NONE"); sys.exit(0)
    print(sub.loc[sub["val_mae"].idxmin(), "$col"])
except Exception:
    print("NONE")
EOF
}

# =============================================================
# Phase 0 — Window Size 결정 (edge=A-fb, features=base)
# =============================================================
phase0() {
    sep
    log "Phase 0: Window Size 그리드서치 (edge=A-fb, features=base)"
    sep
    for LB in 18 36 72 144; do
        run --edge A-fb --lookback "$LB" --features base
    done
}

# =============================================================
# Phase 1 — 엣지 설계 비교 E0~E5 (adaptive 없음)
# =============================================================
phase1() {
    local BEST_LB
    BEST_LB=$(best_from_csv "lookback" \
        "edge=='A-fb' and features=='base' and adaptive=='none'")
    if [ "$BEST_LB" = "NONE" ]; then
        log "ERROR: Phase 0 결과 없음. Phase 0 먼저 실행하세요."
        exit 1
    fi
    BEST_LB=$($PYTHON -c "print(int(float('$BEST_LB')))")

    sep
    log "Phase 1: 엣지 설계 비교 E0~E5 (best_lb=${BEST_LB}, features=base)"
    sep

    run --edge identity --lookback "$BEST_LB" --features base   # E0
    run --edge A-fw     --lookback "$BEST_LB" --features base   # E1
    run --edge A-fb     --lookback "$BEST_LB" --features base   # E2
    run --edge B        --lookback "$BEST_LB" --features base   # E3
    run --edge C        --lookback "$BEST_LB" --features base   # E4
    run --edge D        --lookback "$BEST_LB" --features base   # E5
}

# =============================================================
# Phase 2 — Adaptive 비교 (A0=재사용 / A1=apt / A2=apt+base)
# =============================================================
phase2() {
    local BEST_LB BEST_EDGE
    BEST_LB=$(best_from_csv "lookback" \
        "edge=='A-fb' and features=='base' and adaptive=='none'")
    BEST_LB=$($PYTHON -c "print(int(float('$BEST_LB')))" 2>/dev/null || echo "72")

    BEST_EDGE=$(best_from_csv "edge" \
        "features=='base' and lookback==${BEST_LB} and adaptive=='none' and edge!='identity'")
    [ "$BEST_EDGE" = "NONE" ] && BEST_EDGE="A-fb"

    sep
    log "Phase 2: Adaptive 비교 (best_edge=${BEST_EDGE}, best_lb=${BEST_LB}, features=base)"
    log "  A0 (base only) = Phase 1 best 결과 재사용 — 추가 실행 없음"
    sep

    # A1: Adaptive only (predefined 없음)
    log "▶ A1: Adaptive only"
    run --edge "$BEST_EDGE" --lookback "$BEST_LB" --features base \
        --aptonly --addaptadj --randomadj

    # A2: best_edge + Adaptive
    log "▶ A2: ${BEST_EDGE} + Adaptive"
    run --edge "$BEST_EDGE" --lookback "$BEST_LB" --features base \
        --addaptadj
}

# =============================================================
# Phase 3 — 노드 피처: 해발고도
# =============================================================
phase3() {
    local BEST_LB BEST_EDGE
    BEST_LB=$(best_from_csv "lookback" \
        "edge=='A-fb' and features=='base' and adaptive=='none'")
    BEST_LB=$($PYTHON -c "print(int(float('$BEST_LB')))" 2>/dev/null || echo "72")

    BEST_EDGE=$(best_from_csv "edge" \
        "features=='base' and lookback==${BEST_LB} and adaptive=='none' and edge!='identity'")
    [ "$BEST_EDGE" = "NONE" ] && BEST_EDGE="A-fb"

    sep
    log "Phase 3: 해발고도 노드 피처 (best_edge=${BEST_EDGE}, best_lb=${BEST_LB})"
    sep
    run --edge "$BEST_EDGE" --lookback "$BEST_LB" --features elev
}

# =============================================================
# Phase 4 — 강수량 피처
# =============================================================
phase4() {
    local BEST_LB BEST_EDGE
    BEST_LB=$(best_from_csv "lookback" \
        "edge=='A-fb' and features=='base' and adaptive=='none'")
    BEST_LB=$($PYTHON -c "print(int(float('$BEST_LB')))" 2>/dev/null || echo "72")

    BEST_EDGE=$(best_from_csv "edge" \
        "features=='base' and lookback==${BEST_LB} and adaptive=='none' and edge!='identity'")
    [ "$BEST_EDGE" = "NONE" ] && BEST_EDGE="A-fb"

    sep
    log "Phase 4: 강수량 피처 (best_edge=${BEST_EDGE}, best_lb=${BEST_LB})"
    sep
    run --edge "$BEST_EDGE" --lookback "$BEST_LB" --features rf
}

# =============================================================
# 실행
# =============================================================
PHASE="${1:-all}"

# 전체 실행 시 기존 summary.csv가 있으면 경고
if [ "$PHASE" = "all" ] || [ "$PHASE" = "phase0" ]; then
    if [ -f "results/summary.csv" ]; then
        log "WARNING: results/summary.csv 가 이미 존재합니다."
        log "         이전 실험 결과가 best_lb/best_edge 선택에 영향을 줄 수 있습니다."
        log "         초기화하려면: mv results/summary.csv results/summary_backup.csv"
        log "         5초 후 계속합니다... (Ctrl+C 로 중단)"
        sleep 5
    fi
fi

case "$PHASE" in
    phase0) phase0 ;;
    phase1) phase1 ;;
    phase2) phase2 ;;
    phase3) phase3 ;;
    phase4) phase4 ;;
    all)
        phase0
        phase1
        phase2
        phase3
        phase4
        sep
        log "모든 실험 완료. 결과: results/summary.csv"
        ;;
    *)
        echo "Usage: $0 [phase0|phase1|phase2|phase3|phase4|all]"
        exit 1
        ;;
esac
