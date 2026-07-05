# Graph WaveNet 기반 안성천 수위 예측

안성천 유역 11개 수위 관측소에 대해 **Graph WaveNet** (시공간 그래프 신경망)을 적용한 다중 스텝 수위 예측 연구입니다. 10분~3시간 후(1~18 스텝)를 동시에 예측하며, 그래프 위상·적응형 인접행렬·노드 피처·강수량을 단계별로 비교하는 5단계 절제 실험을 수행합니다.

> [Graph WaveNet for Deep Spatial-Temporal Graph Modeling (IJCAI 2019)](https://arxiv.org/abs/1906.00121) 기반 구현

---

## 주요 결과

| 설정 | Test MAE (m) | Test RMSE (m) | NSE |
|---|:---:|:---:|:---:|
| E0 — identity (그래프 없음) | 0.0128 | 0.0530 | 0.9946 |
| E2 — 수계 위상 A-fb | 0.0113 | 0.0489 | 0.9955 |
| E2 + 강수량 (최근접) | **0.0111** | **0.0464** | **0.9959** |

수계 위상 기반 방향 그래프(E2)가 그래프 없는 기준선 대비 MAE **11.7%** 개선.  
강수량 피처를 추가하면 MAE 추가 **1.8%**, RMSE **5.1%** 개선.

---

## 연구 대상

- **유역**: 안성천 (경기·충남 경계)
- **관측소**: HRFCO 수위 관측소 11개소
- **시간 해상도**: 10분 간격
- **데이터 분할**: 훈련 2019–2022 · 검증 2023 · 테스트 2024

---

## 파일 구조

```
Graph-WaveNet/
├── model.py                  # Graph WaveNet 아키텍처
├── engine.py                 # 학습/평가 루프 (Trainer 클래스)
├── train.py                  # 학습 진입점
├── eval.py                   # 테스트셋 평가
├── util.py                   # 스케일러, 지표 (MAE, RMSE, MAPE, NSE)
├── graph_utils.py            # 인접행렬 생성 (E0~E5, 수계 위상/거리/상관)
├── generate_training_data.py # 슬라이딩 윈도우 .npz 생성
├── run_experiments.sh        # Phase 0~4 전체 자동 실행 스크립트
└── slide.pdf                 # 발표 자료
```

> **본 저장소는 코드 열람용입니다.** 관측 데이터·관측소 메타데이터(`api/`), 학습 데이터(`data/`), 체크포인트(`garage/`), 실험 결과(`results/`), EDA(`eda/`)는 저장소에 포함되지 않습니다. 따라서 클론 후 곧바로 실행되지는 않으며, 실행하려면 아래 *데이터 준비* 절차대로 데이터를 갖추어야 합니다.

---

## 환경 설정

```bash
conda create -n water-level python=3.10
conda activate water-level
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scipy
```

---

## 데이터 준비

> 관측 데이터와 관측소 메타데이터(`api/`)는 저장소에 포함되지 않으므로, 아래 파일을 직접 준비해야 합니다.

`generate_training_data.py`는 다음 파일을 입력으로 사용합니다.

- `api/waterlevel_info.json` — 관측소 메타데이터 (이름·좌표·영점표고). `graph_utils.py`가 인접행렬 생성에 사용
- `api/data/waterlevel_10m_filled.csv` — 결측 보간된 10분 수위 (11개소)
- `api/data/rainfall_mapped_10m.csv` — 관측소별 최근접 강수량 (`rf` 피처 사용 시)

수위·강수량 원시 자료는 [HRFCO Open API](https://www.hrfco.go.kr)에서 수집합니다. 위 파일이 준비되면 슬라이딩 윈도우 .npz로 변환합니다.

```bash
# 기본 피처 (수위 + 시각 정보, lookback=36스텝)
python generate_training_data.py --lookback 36 --features base
```

출력: `data/anseong/lb{lookback}_{features}/{train,val,test}.npz`

피처 옵션: `base` (수위+시각) · `elev` (+해발고도) · `rf` (+강수량 최근접)

---

## 학습

```bash
# 단일 실험
python train.py --edge A-fb --lookback 36 --features rf

# Phase 0~4 전체 절제 실험
bash run_experiments.sh

# 특정 단계만 실행
bash run_experiments.sh phase1   # 엣지 설계 비교 (E0~E5)
bash run_experiments.sh phase4   # 강수량 피처 비교
```

**주요 인자**

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--edge` | `A-fb` | `identity` / `A-fw` / `A-fb` / `B` / `C` / `D` |
| `--lookback` | `72` | 입력 시퀀스 길이 (스텝) |
| `--features` | `base` | `base` / `elev` / `rf` |
| `--addaptadj` | — | 적응형 인접행렬 추가 학습 |
| `--aptonly` | — | 적응형 인접행렬만 사용 |
| `--epochs` | `100` | 최대 에폭 |
| `--patience` | `15` | 조기 종료 patience (val MAE 기준) |
| `--seed` | `42` | 랜덤 시드 |

체크포인트는 `garage/anseong/<태그>/`에 저장됩니다.

---

## 평가

```bash
python eval.py --edge A-fb --lookback 36 --features rf
```

MAE, RMSE, MAPE, NSE를 horizon 스텝별 및 전체 집계로 출력합니다.

---

## 절제 실험 요약

| 단계 | 질문 | 결론 |
|---|---|---|
| Phase 0 | 최적 Lookback 길이? | **36스텝 (6h)** |
| Phase 1 | 최적 엣지 설계? | **E2: 방향 위상 (A-fb)** |
| Phase 2 | 적응형 인접행렬 효과? | **사전 정의 + 적응형 결합** |
| Phase 3 | 해발고도 피처 효과? | 없음 — 위상에 이미 내포 |
| Phase 4 | 강수량 피처 효과? | **유효 — RMSE 개선 (홍수 피크)** |

---

## 인용

```bibtex
@inproceedings{wu2019graph,
  title     = {Graph WaveNet for Deep Spatial-Temporal Graph Modeling},
  author    = {Wu, Zonghan and Pan, Shirui and Chen, Fengwen and Long, Guodong and Zhang, Chengqi and Yu, Philip S.},
  booktitle = {Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI)},
  year      = {2019}
}
```
