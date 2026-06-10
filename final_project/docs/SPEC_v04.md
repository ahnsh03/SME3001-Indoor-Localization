# SPEC v04 — v03 + HardGate (기하 잔차) [DEPRECATED]

> **기말 프로젝트에서는 사용하지 않습니다.** 실측 UWB 이상 대응용 중간 기법.

## 추가 기법 (v03 대비)

**단 1가지:** 중간 V15식 **HardGate**

1. 보정 거리 \(\tilde{\mathbf{d}}\)로 Huber 삼변 → \(\mathbf{p}_0\).
2. \(r_k = \big|\|\mathbf{p}_0-\mathbf{a}_k\| - \tilde{d}_k\big|\).
3. \(r_k > \tau\) 인 BS 제거 후 Huber 재삼변.

**τ 선택:** Train 5-Fold OOF 잔차 풀 → 분위수 50~95% 후보 그리드 → **CV RMSE 최소** τ.

## 설정 (본 실행)

| 항목 | 값 |
|------|-----|
| `calib` | `per_bs_affine` |
| `gate` | on |
| `gate_tau` | **26.76** (grid best) |
| `loss` | `huber` |

## CV 결과

| 지표 | v03 | v04 | Δ |
|------|-----|-----|---|
| OOF RMSE (m) | 12.52 | **12.78** | **+0.26** (악화) |

스테이지:

| 스테이지 | RMSE (m) |
|----------|----------|
| after_main_trilat | 12.52 |
| after_gate | **12.78** |

## 교훈 (보고서용)

- 실측 UWB에서 게이트가 유효했던 것과 달리, **합성 RTT+아핀 보정 후**에는 채널 제거가 **정보 손실**을 유발.
- τ가 크게(26.76 m) 잡혀도 CV는 v03보다 나빠짐 → **기말 production에 게이트 미포함**.
- 향후: τ 그리드 축소, soft gate, 보정 **후** 잔차 분포 재분석 후 재시도.

## gate_tuning 산출

`outputs/v04_summary.json` 내 `gate_tuning.grid` — τ별 CV RMSE 기록.

## 산출물

- `outputs/v04_summary.json`

## 실행

```powershell
py -3 scripts/run_all_versions.py --versions v04
```
