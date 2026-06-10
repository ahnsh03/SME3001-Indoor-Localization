# SME3001 Indoor Localization

스마트모빌리티공학실험2 **중간·기말 프로젝트** 통합 저장소입니다.

| 디렉터리 | 내용 | 실행 루트 |
|----------|------|-----------|
| [`midterm_project/`](midterm_project/) | Wi‑Fi & UWB 융합 실내 측위 (V1~V16 파이프라인) | `midterm_project/` |
| [`final_project/`](final_project/) | 셀룰러 RTT 기반 2D 측위 (개발·실험, production **v30**) | `final_project/` |

## 빠른 시작

### 중간 프로젝트 (V15 권장)

```powershell
Set-Location ".\midterm_project"
py -3 scripts\indoor_fusion_pipeline_v15.py
```

상세: [`midterm_project/README.md`](midterm_project/README.md)

### 기말 프로젝트

```powershell
Set-Location ".\final_project"
py -3 main.py   # data/InF_DH_FR1.mat 포함 (README 참고)
```

과제 규격: [`final_project/final_project_discription.md`](final_project/final_project_discription.md)

## 저장소 구조

```text
.
├── README.md                 # 이 파일
├── midterm_project/
│   ├── data/
│   ├── docs/
│   ├── scripts/
│   └── outputs/
└── final_project/
    ├── data/           # InF_DH_FR1.mat
    ├── docs/           # 11주차 연구 계획 보고서
    ├── scripts/
    └── scripts/, lib/, outputs/  (제출본은 별도 repo)
```

## Git

```text
origin  https://github.com/ahnsh03/SME3001-Indoor-Localization.git
branch  main
```

기존 GitHub 이력은 **루트에 있던 midterm**을 `midterm_project/`로 옮긴 모노레포 구조로 이어집니다.

**기말 과제 제출 저장소 (별도):** [SME-Experiment2-FinalProject-12223637](https://github.com/ahnsh03/SME-Experiment2-FinalProject-12223637)
