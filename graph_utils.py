"""
안성천 관측소 인접행렬 생성 — Graph WaveNet 전용

실험별 엣지 설계:
  identity : 연결 없음 (공간정보 제거, 기준선)
  A-fw     : 수계 위상 directed, forward-only   → [Pf]
  A-fb     : 수계 위상 directed, forward+backward → [Pf, Pb]
  B        : 수계 위상 undirected               → [sym]
  C        : 지리적 거리 Gaussian undirected     → [sym]
  D        : Pearson 상관 undirected            → [sym]

Adaptive-only / Best+Adaptive 는 train.py 에서 --aptonly / --addaptadj 로 제어.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from pathlib import Path

# ── 관측소 코드 (수위파일 컬럼 순서와 동일) ───────────────────
WL_CODES = [
    "1101605",  # 화성시(화산교)
    "1101610",  # 안성시(옥산대교)
    "1101620",  # 안성시(건천리)
    "1101630",  # 천안시(안성천교)
    "1101635",  # 평택시(군문교)
    "1101645",  # 오산시(탑동대교)
    "1101650",  # 화성시(수직교)
    "1101663",  # 평택시(진위1교)
    "1101665",  # 평택시(회화리)
    "1101670",  # 평택시(동연교)
    "1101680",  # 평택시(팽성대교)
]

_WL_INFO_PATH = Path(__file__).parent / "api" / "waterlevel_info.json"


def _dms_to_dd(dms: str) -> float:
    d, m, s = dms.strip().split("-")
    return float(d) + float(m) / 60 + float(s) / 3600


def _load_wl_meta() -> dict:
    """waterlevel_info.json → {code: (name, lat, lon, gdt)} — WL_CODES 순서."""
    import json
    with open(_WL_INFO_PATH) as f:
        rows = json.load(f)["content"]
    lookup = {r["wlobscd"]: r for r in rows}
    result = {}
    for cd in WL_CODES:
        r = lookup[cd]
        result[cd] = (
            r["obsnm"],
            _dms_to_dd(r["lat"]),
            _dms_to_dd(r["lon"]),
            float(r["gdt"]),
        )
    return result


_WL_META = _load_wl_meta()

STATIONS  = [_WL_META[cd][0] for cd in WL_CODES]
N         = len(STATIONS)
# 좌표 출처: waterlevel_info.json lat/lon (DMS → 십진수)
COORDS    = np.array([[_WL_META[cd][1], _WL_META[cd][2]] for cd in WL_CODES], dtype=np.float32)
# 해발고도 출처: waterlevel_info.json gdt (영점표고, m)
ELEVATION = np.array([_WL_META[cd][3] for cd in WL_CODES], dtype=np.float32)

# 수계 위상 엣지 (상류 → 하류)
BINARY_EDGES = [
    (1, 2),   # 옥산대교 → 건천리
    (2, 3),   # 건천리   → 안성천교
    (3, 4),   # 안성천교 → 군문교
    (4, 10),  # 군문교   → 팽성대교
    (0, 6),   # 화산교   → 수직교
    (6, 9),   # 수직교   → 동연교
    (5, 8),   # 탑동대교 → 회화리
    (7, 8),   # 진위1교  → 회화리
    (8, 9),   # 회화리   → 동연교 (합류)
    (9, 10),  # 동연교   → 팽성대교
]

_WL_PATH   = Path(__file__).parent / "api" / "data" / "waterlevel_10m_filled.csv"
_TRAIN_END = "2022-12-31 23:50"


# ── 정규화 유틸 (util.py 와 동일 로직, 의존성 분리용) ─────────

def _asym_adj(adj: np.ndarray) -> np.ndarray:
    """행 정규화: D^-1 A  (비대칭, forward diffusion)"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    with np.errstate(divide="ignore"):
        d_inv = np.where(rowsum == 0, 0.0, 1.0 / rowsum)
    return sp.diags(d_inv).dot(adj).astype(np.float32).toarray()


def _sym_adj(adj: np.ndarray) -> np.ndarray:
    """대칭 정규화: D^-1/2 A D^-1/2"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d = sp.diags(d_inv_sqrt)
    return adj.dot(d).T.dot(d).astype(np.float32).toarray()


# ── 원시 인접행렬 빌더 ────────────────────────────────────────

def _topology_directed() -> np.ndarray:
    """수계 위상 directed 이진 행렬 (비대칭, 상류→하류)."""
    A = np.zeros((N, N), dtype=np.float32)
    for i, j in BINARY_EDGES:
        A[i, j] = 1.0
    return A


def _topology_undirected() -> np.ndarray:
    """수계 위상 undirected 이진 행렬 (대칭)."""
    A = _topology_directed()
    return np.maximum(A, A.T)


def _distance_gaussian(threshold: float = 0.1) -> np.ndarray:
    """
    지리적 거리 기반 Gaussian 가중치 행렬 (무방향).
    w_ij = exp(-(d_ij / σ)^2),  σ = 거리의 표준편차
    threshold 미만 값은 0으로 처리.
    """
    lat_km = 111.0   # km / degree lat
    lon_km = 88.5    # km / degree lon (37° 기준)
    coords_km = COORDS * np.array([lat_km, lon_km])

    dist = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            dist[i, j] = np.linalg.norm(coords_km[i] - coords_km[j])

    sigma = dist[dist > 0].std()
    W = np.exp(-(dist / sigma) ** 2)
    W[W < threshold] = 0.0
    np.fill_diagonal(W, 0.0)
    return W.astype(np.float32)


def _pearson_correlation(threshold: float = 0.5) -> np.ndarray:
    """
    훈련 데이터(2019-2022) Pearson 상관 기반 행렬 (무방향).
    |r| < threshold 는 0으로 처리.
    """
    df = pd.read_csv(_WL_PATH, index_col=0, parse_dates=True)
    train = df[df.index <= _TRAIN_END][STATIONS]
    C = train.corr().abs().values.astype(np.float32)
    C[C < threshold] = 0.0
    np.fill_diagonal(C, 0.0)
    return C


# ── 메인 API ──────────────────────────────────────────────────

def get_supports(
    edge_type: str,
    device: str | torch.device = "cpu",
    corr_threshold: float = 0.5,
    dist_threshold: float = 0.1,
) -> list[torch.Tensor]:
    """
    edge_type에 따른 GWN support 행렬 리스트 반환.

    Args:
        edge_type       : 'identity' | 'A-fw' | 'A-fb' | 'B' | 'C' | 'D'
        device          : torch device
        corr_threshold  : Edge D에서 |r| 기준 컷오프 (default 0.5)
        dist_threshold  : Edge C에서 Gaussian 기준 컷오프 (default 0.1)

    Returns:
        List of (N, N) float32 Tensor — gwnet 의 supports 인자로 직접 사용
        Adaptive-only 실험은 빈 리스트([]) 또는 None 을 넘기고 --aptonly 플래그 사용.
    """
    if edge_type == "identity":
        mats = [np.eye(N, dtype=np.float32)]

    elif edge_type == "A-fw":
        A = _topology_directed()
        mats = [_asym_adj(A)]

    elif edge_type == "A-fb":
        A = _topology_directed()
        mats = [_asym_adj(A), _asym_adj(A.T)]

    elif edge_type == "B":
        A = _topology_undirected()
        mats = [_sym_adj(A)]

    elif edge_type == "C":
        W = _distance_gaussian(threshold=dist_threshold)
        mats = [_sym_adj(W)]

    elif edge_type == "D":
        C = _pearson_correlation(threshold=corr_threshold)
        mats = [_sym_adj(C)]

    else:
        raise ValueError(f"Unknown edge_type '{edge_type}'. "
                         f"Choose from: identity, A-fw, A-fb, B, C, D")

    return [torch.tensor(m, dtype=torch.float32).to(device) for m in mats]


def num_supports(edge_type: str) -> int:
    """
    edge_type 별 support 행렬 개수.
    gwnet 초기화 시 support_len 계산에 사용.
    (Adaptive adj 추가 시 +1 은 gwnet 내부에서 처리)
    """
    return 2 if edge_type == "A-fb" else 1
