# 기말 프로젝트 — 셀룰러 RTT 2D 측위

## Production: **v30** (제출: `submit/` 별도 repo)

| 항목 | 값 |
|------|-----|
| 보정 | **Isotonic + MLP** 거리 보정 |
| 삼변 | **비대칭 Huber** + soft weight |
| 후처리 | **2-pass 위치 아핀** |
| CV OOF RMSE | **~8.0 m** (v00 22.4 m 대비) |

문서: [`docs/DEV_DECISION_LOG.md`](docs/DEV_DECISION_LOG.md) · [`docs/SPEC_v30.md`](docs/SPEC_v30.md)

## 실행

```powershell
Set-Location ".\final_project"
py -3 scripts/analyze_for_decisions.py   # 관찰 JSON
py -3 scripts/run_all_versions.py
py -3 train.py --version v30
py -3 main.py
```

## 기록·산출물

| 파일 | 내용 |
|------|------|
| `outputs/decision_analysis.json` | D1~D6 관찰 수치 |
| `outputs/version_catalog.json` | CV 비교 |
| `outputs/vXX_summary.json` | fold·스테이지 로그 |
| `docs/DEV_DECISION_LOG.md` | 관찰→판단→버전 |

HardGate(v04) **미사용**.
