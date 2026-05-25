# 기말 프로젝트 — 셀룰러 RTT 2D 측위

인공 생성 데이터(`InF_DH_FR1.mat`, 채점 시 `DH_FR1.mat`)로 18개 기지국 RTT 기반 사용자 위치를 추정합니다.

## 제출물 (repo 루트 = `final_project/`)

| 파일 | 설명 |
|------|------|
| `main.py` | 추론 (채점기 실행) |
| `train.py` | 학습 로직 (비ML은 `main.py`와 동일 복사 가능) |
| `report.md` | 보고서 |
| `requirements.txt` | 표준 환경 외 패키지 사용 시만 |

자세한 규격: [`final_project_discription.md`](final_project_discription.md)

## 디렉터리

| 경로 | 내용 |
|------|------|
| `data/InF_DH_FR1.mat` | 기말 학습용 합성 RTT 데이터 (700 UE) |
| `docs/` | 11주차 연구 계획 보고서(구 `11주차 보고서/`), EDA 그림·JSON |
| `scripts/` | Week 11 EDA·Figure 생성 (`week11_*.py`) |

## 데이터

```text
final_project/data/InF_DH_FR1.mat
```

채점기는 동일 내용을 작업 디렉터리에 `DH_FR1.mat` 이름으로 배치합니다. 로컬 스크립트는 `data/InF_DH_FR1.mat`을 읽습니다.

## 실행 (개발 중)

```powershell
Set-Location ".\final_project"
py -3 scripts\week11_final_eda_deep.py
```

`main.py` / `train.py`는 기말 제출 형식에 맞춰 추가 예정입니다.
