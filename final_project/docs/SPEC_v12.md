# SPEC v12 — 반경 3구간 Isotonic

## 관찰 (D6)

near/mid/far bin에서 global 보정 후 RMSE **7.4 / 9.1 / 14.0 m** (`decision_analysis.json`) → **공간별 보정 분리** 가설.

## 추가 기법

1. Train fold UE 반경 \(r=\|p-c\|\), 구간: **near <15 m**, **mid 15–30 m**, **far ≥30 m**.
2. BS×구간별 Isotonic (표본 <25면 fold 전체 fallback).
3. 추론 2-pass: **fallback Isotonic** → Huber → \(r\) → 구간 Isotonic → Huber.

HardGate **미사용**.

## CV 결과

| | v06 | v12 | Δ |
|---|-----|-----|---|
| OOF RMSE (m) | 10.51 | 10.23 | −0.28 |

초기 구현(mid-zone 1-pass)은 **22.2 m**으로 실패 → fallback pass1 수정 후 10.23 m.

## 판단

v06·v11 대비 **이득 없음** — 구간별 Isotonic은 보고서 ablation만, production **미채택**.

## 버그 교훈

| 이슈 | 수정 |
|------|------|
| pass1에 mid 구간만 사용 | `apply_fallback()` 전체 train Isotonic |

기록: `DEV_DECISION_LOG.md` §2.
