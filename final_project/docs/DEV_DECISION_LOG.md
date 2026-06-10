# 개발·판단 기록 (Data → Decision → Version)

> HardGate **미사용**. 갱신: **2026-06-02** (RMSE 4 m 목표 — anchor-WLS + 블렌드 실험)

---

## 0. 공간 정의 (`lib/spatial.py`)

| 표기 | 값 |
|------|-----|
| 설계 반폭 | **±60 m (x) × ±30 m (y)** |
| 전체 span | **120 m × 60 m** |
| 삼변 bounds | **[-62,62] × [-32,32]** (반폭+2 m margin) |
| BS 배치 span | 100 m × 40 m |

측정 UE: x span 119.99 m, y span 59.68 m → 설계와 일치.

| Tier | RMSE 목표 | 상태 (v17 CV) |
|------|-----------|----------------|
| G0 baseline | < 15 m | ✓ |
| G1 기말 합격 | < 10 m | ✓ **8.90 m** |
| G2 양호 실내 | < 7 m | ✗ (8.90 m) |
| G3 UWB급 | < 5 m | ✗ (데이터 +16 m bias 한계) |

---

## 1. 데이터 관찰

| ID | 관찰 | 수치 |
|----|------|------|
| D1 | d_true–d̂ 단조 | Pearson **0.815** |
| D2 | NLOS 양의 bias | 잔차 mean **+15.9 m**, 양의 비율 **81%** |
| D3 | 전역 아핀 | α≈0.61, β≈9.2 |
| D4 | BS별 아핀 CV 악화 | v03 +0.11 m |
| D5 | 2차 보정 | full-fit Huber **11.4 m** |
| D6 | 공간 불균형 | near **7.4** / mid **9.1** / far **14.0 m** (global 보정 후) |
| D8 | Huber > LS | 16.7 vs 22.4 m |
| D10 | HardGate | **폐기** (+0.26 m) |

재생성: `py -3 scripts/analyze_for_decisions.py`, `py -3 scripts/eda_3gpp_channel.py`

---

## 2. 누수·검증 감사 (`outputs/leakage_audit.json`)

| 검사 | 결과 | 비고 |
|------|------|------|
| `main.py` 추론 | ✓ | GT는 로깅 전용, 예측 경로 미사용 |
| fold별 calib 분리 | ✓ | val UE에서 train-fit vs fold-fit 보정 차이 **> 7 m** |
| MLP train index | ✓ | `train_idx`만으로 통계·학습 |
| 하이퍼파라미터 튜닝 | ⚠ medium | γ·f_scale·blend는 **전체 700 OOF**로 선택 → 약간 낙관적 |
| 완화 | holdout **560/140** | `outputs/holdout_eval.json` |
| holdout vs CV (v16) | CV(train만) **8.79 m**, holdout **9.31 m**, gap **+0.52 m** | 과적합 경계 양호 |

**수정·개선 (이번 라운드)**

- 삼변 **bounds**를 UE span+2 m으로 조정, CV 시 **fold train 위치**로 bounds 산출 (추론 시 고정 bounds).
- `top_k`는 **Isotonic 보정 후** shortest-k 적용 (이전: 보정 전 잘라 18채널 불일치 버그 수정).

---

## 3. 버전 타임라인 → CV OOF RMSE (m)

| ver | 기법 | CV RMSE | Δ vs v06 | 채택 |
|-----|------|---------|----------|------|
| v00 | LS | 22.42 | — | |
| v01 | Huber | 16.71 | | |
| v02 | 전역 아핀 | 12.40 | | |
| v06 | Isotonic | 10.51 | — | |
| v11 | +soft γ=1.5 | 9.77* | −0.74 | |
| **v16** | Isotonic+MLP | **9.28**† | −1.23 | |
| **v17** | v16+γ=**1.0** | **8.90**† | −1.61 | **★ production** |
| v13 | MLP만 | 10.81 | | ✗ |
| v14 | 비대칭 Huber | 10.33 | | ✗ |
| v15 | shortest-k | 10.51 | | ✗ |
| v18 | PosMLP 직접 | 38.06 | | ✗ (700장 과적합) |
| v19 | GBR 거리 | 11.10 | | ✗ |
| v20 | v16+Pos blend | 10.19 | | ✗ |

\* bounds 조정 전 수치  
† bounds 조정 후 재측정

| v25 | 2-pass 위치 아핀 | 8.76 | −0.14 | |
| **v30** | **v25+asym w=5** | **7.996** | **−0.77** | **★ production** |
| v26 | far 2차 Isotonic | 9.28 | +0.38 | ✗ |
| v27 | quantile τ CV | 9.61 (τ=0.35) | +0.71 | ✗ |

**Production:** `v30` — `py -3 train.py --version v30`  
산출: `config.json`, `model_calib.npz`, `model_mlp.pt`, `pos_affine`

---

## 4. 스테이지 분해 (v17)

| 스테이지 | RMSE (m) |
|----------|----------|
| after_raw_trilat | ~16.7 |
| after_main_trilat | **8.90** |

Isotonic+MLP 보정 후 γ=1.0 soft weight로 추가 **~0.38 m** (v16 대비).

---

## 5. 실패·교훈

| 버전 | 교훈 |
|------|------|
| v18 | d̂→(x,y) 직접 회귀는 CV에서 **38 m** — 라벨 공간 과적합, geometry 무시 |
| v20 | trilat+pos 블렌드는 v17보다 나쁨 — 두 해가 상충 |
| v19 | GBR는 MLP 2단계보다 열세 |
| G2 (<7 m) | far 구간(**14 m**) 한계; 전역 RMSE **8.9 m**이 현실적 상한에 근접 |

---

## 6. 꼬리·far UE 분석 (`outputs/tail_analysis.json`, v17)

| 구간 (GT 반경) | n | RMSE | median |
|----------------|---|------|--------|
| near | 96 | 6.56 m | 5.50 m |
| mid | 249 | 8.68 m | 6.75 m |
| far | 355 | **9.58 m** | 7.49 m |

전체 median **6.88 m** vs RMSE **8.90 m** → **far/꼬리가 RMSE 견인**.

---

## 7. 구조 변경 3종 (`outputs/structural_v25_v27.json`)

| ver | 내용 | RMSE | 판단 |
|-----|------|------|------|
| v25 | 1차 삼변 후 **위치 아핀** (train만 피팅) | **8.76** | **채택** — p90도 개선 |
| v26 | far UE **2차 Isotonic** | 9.28 | 거리 보정 이중화는 이득 없음 |
| v27 | **quantile τ** 0.25–0.45 CV | 9.61 | 전 구간에서 Isotonic+MLP 열세 |

---

## 8. 단일 요인 ablation (v17 대비, `outputs/ablation_results.json`)

| ver | 가설 | CV RMSE | Δ vs v17 | 판단 |
|-----|------|---------|----------|------|
| v17 | baseline | **8.90** | — | **유지** |
| v21 | min(d̂) 3구간 Isotonic+MLP | 8.91 | +0.01 | ✗ |
| v22 | quantile τ=0.35 + MLP | 9.61 | +0.71 | ✗ |
| v23 | raw d̂ zone 가중 (near↑ far↓) | 8.90 | 0 | ✗ (채널별 시도 시 9.25로 악화) |
| v24 | MLP 노이즈 증강 σ=0.03 | 9.07 | +0.17 | ✗ |

**구조적 한계:** d̂−d_true 양의 bias(+16 m) + 삼변 기하 → RMSE **~9 m** 근처가 현 파이프라인 상한. median은 이미 **~6.9 m**.

---

## 9. 조합 재검토 (`outputs/combination_review.json`)

| ver | 조합 | RMSE | Δ v25 | 판단 |
|-----|------|------|-------|------|
| v28 | γ CV (MLP+아핀) | 8.76 | 0 | best γ=**1.0** (v11의 1.5는 MLP 없을 때만 유리) |
| v29 | f_scale CV | 8.67 | −0.09 | 소폭 개선, v30보다 열세 |
| **v30** | **asym CV** | **7.996** | **−0.77** | **v14 단독 실패→스택 시 대성공** |
| v31 | top_k | 8.76 | 0 | 18채널 필수 |
| v32 | γ=1.5 고정 | 8.85 | +0.09 | MLP와 조합 시 무효 |
| v33 | γ1.5+f0.8 | 8.90 | +0.14 | 동일 |
| v34 | far 2차 아핀 | 8.76 | 0 | 무효 |
| v35 | mindist+아핀 | 8.75 | −0.01 | 무효 |

**놓쳤던 통찰:** 비대칭 Huber는 **거리 보정·아핀 이후 삼변**에 써야 NLOS(+bias)와 맞는다.

---

## 10. 재현 명령

---

## 9. RMSE 4 m 추진 (2026-06-02, `submit/` 미반영)

| 실험 | OOF RMSE (m) | 비고 |
|------|--------------|------|
| v30 (production) | **7.996** | 1차 채점 ~7.99와 일치 |
| end2end anchor-WLS (best holdout 5.545) | **6.343** | holdout→OOF 갭 있음 |
| v30 + anchor grid blend (α=0.20) | **6.214** | `exp_blend_v30_anchor_oof.py` |
| oracle (v30 vs anchor, OOF) | **5.185** | 샘플별 최선 선택 하한 |
| anchor on v30-calib only | 7.675 | 거리 스케일 학습 제거 시 악화 |
| anchor v2 (mixup/focal) | 11.483 | 폐기 |
| physics stack (v30calib+affine) | 7.574 | v30 단독보다 낮지 않음 |

**판단:** 4 m은 현재 700명 OOF·후보군 기준으로 **oracle 하한 ~5.2 m** — 단일 모델·누수 없는 블렌드만으로는 4 m 진입이 어렵다. 다음 후보: 구역별 α(누수 없는 fold 내 튜닝), anchor 장기 학습(360 epoch), soft gate 메타.

재현:
```powershell
py -3 scripts/exp_end2end_anchor_wls_oof_best.py
py -3 scripts/exp_blend_v30_anchor_oof.py
py -3 scripts/exp_oracle_oof_floor.py
py -3 scripts/exp_v40_hlos_oof.py
```

### v40 (2026-06-02): v30 비의존 HLOS 파이프라인
- **입력:** raw `d_hat` (v30 Isotonic+MLP 제거)
- **1)** train-fold per-BS affine + MAD outlier trim
- **2)** LOS MLP (상대 feature, abs 라벨 `|d-d_true|<2m`)
- **3)** Rwgh subset (top-8, k=4) + P(LOS) 가중 NLS
- **4)** HGB+ExtraTrees on `delta=p_true-p_geo` (geometry+잔차 feature)
- **5)** 2-pass 위치 아핀 (train coarse만)
- holdout best: raw+abs HLOS **3.28 m**; OOF **5.13 m** (fold 편차 큼)

```powershell
cd final_project
py -3 scripts/audit_leakage.py
py -3 scripts/spatial_goals.py
py -3 scripts/analyze_tail.py
py -3 scripts/run_ablation.py
py -3 scripts/run_structural_v25_v27.py
py -3 scripts/run_combination_review.py
py -3 scripts/holdout_eval.py
py -3 train.py --version v30
py -3 main.py
```

| 산출 | 경로 |
|------|------|
| 누수 감사 | `outputs/leakage_audit.json` |
| 목표 정의 | `outputs/spatial_goals.json` |
| 실험 루프 | `outputs/experiment_loop.json` |
| Hold-out | `outputs/holdout_eval.json` |
| CV 요약 | `outputs/v17_summary.json` |
