# 스마트모빌리티공학실험2 Final Project

## 1. 제출물 (GitHub repo)

Repository 루트에 다음 파일을 저장합니다. iclass에는 URL 만 폼으로 제출. **Public repo 만 허용 (Private 금지).**

| 파일 | 누가 |
|------|------|
| `main.py` | 모든 학생 |
| `train.py` | **모든 학생** (비ML 학생은 `main.py` 와 동일 내용 복붙해서 제출. ML 학생은 학습 로직) |
| `report.md` | 모든 학생 |
| `requirements.txt` | **표준 환경 외 패키지 사용 시만** (§2 참조) |
| `model.*` (.pkl, .pt 등) | ML 사용 학생만 (학습된 가중치, 재현성 확보용) |

> ⚠️ 참신성·일치성 채점은 **`report.md` + `train.py`** 를 기준으로 합니다 (`main.py` 아님).
> 비ML 학생도 `train.py` 필수 — `main.py` 와 같은 내용을 그대로 두 번 제출하시면 됩니다.

---

## 2. 표준 실행 환경

채점기는 다음 버전으로 실행합니다. **이 외 패키지 사용 시 `requirements.txt` 필수**.

| 패키지 | 버전 |
|--------|------|
| Python | 3.12.3 |
| numpy | 2.4.4 |
| scipy | 1.17.1 (`scipy.io`, `scipy.optimize` 포함) |
| pytorch | 2.8.0 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.8 |
| pandas | 2.3.x (보조) |

위 패키지만 쓰면 `requirements.txt` **불필요**. 다른 패키지 사용 시 `pip install -r requirements.txt` 를 사용해서 설치가 가능하도록 `requirements.txt` 를 작성.

---

## 3. 데이터

채점에 쓰이는 `.mat` 파일은 다음 3개 변수를 담고 있습니다.

| 변수 | 모양 | 의미 |
|------|------|------|
| `p`     | (2, N) | 정답 사용자 위치 (x, y) |
| `d_hat` | (18, N) | 18개 기지국이 측정한 RTT |
| `p_bs`  | (2, 18) | 18개 기지국의 좌표 |

전체 사용자 (UE) 는 **1000명**. 그 중 **700명 만 학생에게 제공**, 나머지 **300명** 은 조교가 hidden test set 으로 보유. 채점 시 채점기는 hidden 데이터로 학생 main.py 를 실행합니다 → 학생이 받은 데이터에만 over-fit 한 코드는 손해.

---

## 4. `main.py` 작성 규격

```python
import numpy as np
import scipy.io as sio

def your_algorithm(d_hat[:, u], p_bs):
    """
    본인 알고리즘 작성(필요시 상단에 추가적인 함수 작성 가능)
    """
    return 측위결과

def main():
    # 1) 입력 데이터 로드 — 채점기가 같은 폴더에 .mat 파일 자동 배치
    mat_path  = 'DH_FR1.mat'
    
    data = sio.loadmat(data_path, squeeze_me=False)
    p_bs   = np.asarray(data['p_bs'], dtype=float)     # (2, 18)
    d_hat  = np.asarray(data['d_hat'], dtype=float)    # (18, num_user)
    p      = np.asarray(data['p'], dtype=float)        # (2, num_user) — GT 위치

    # 2) 본인 알고리즘 — 사용자 수는 입력에서 동적으로 받기
    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user))
    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], p_bs)

    # 3) 결과 반환 — numpy 배열, 모양 (2, num_user)
    return p_hat

 if __name__ == "__main__":
    main()
```

⭐ 터미널에서 python main.py 로 실행 가능하도록 작성

### 규칙

| 규칙 | 설명 |
|------|------|
| `main()` 함수 정의 | 채점기가 호출 |
| ⭐ **결과 반환 = numpy 배열, 모양 `(2, num_user)`** | 첫 행 = x 좌표, 둘째 행 = y 좌표 |
| ⭐ **사용자 수를 코드에 미리 박지 말기** | `num_user = d_hat.shape[1]` 로 입력에서 받기 |
| **실행 시간 10분** 제한 | 그 안에 `main()` 이 안 끝나면 강제 종료 → **성능점수 최하점** |
| 파일 이름은 `DH_FR1.mat` | 채점기가 이 이름으로 cwd 에 자동 배치 |

---

## 5. `report.md` 작성 규칙

### 5.1 필수 섹션 (이 4개만)

1. **모티베이션 & 인트로** — 중간발표까지의 실험 결과·고찰 정리, 거기서 본 알고리즘 아이디어가 도출된 흐름, 알고리즘의 high-level 소개.
2. **알고리즘 설명** — 어떻게 동작하는지 구체적으로 설명 (**말과 수식으로만**). **이 설명만 듣고 코드 구현이 가능해야 함.**
3. **Agent AI(e.g., ChatGPT, Claude Code, Gemini 등) 활용 방안** — 어떤 Agent AI를 어떤 방식으로 활용하였는지 구체적으로 작성(AI와 본인의 역할 구분 필요).
4. **결과 도출 & 디스커션** — 수치의 단순 비교 X. 본인의 사고와 구현이 적합했는가 / baseline과의 비교가 fair 한가 (예: 딥러닝 vs 단순 삼각측량 비교는 unfair) / 알고리즘의 장점·단점 / future work / 본인이 사용한 자체 평가 방식의 fairness.
5. **Reference** — 참고한 논문이 있는 경우, 레퍼런스를 달고 해당 논문이 제안하는 부분과 본인이 제안하는 부분의 차이를 2번 파트에 명확히 기재.


### 5.2 형식 제한 (엄격)

| 제한 | 이유 |
|------|------|
| ❌ **코드 블록 (```...```) 금지** | 코드는 `main.py` 로 평가, 보고서는 자연어 설명 평가 |
| ❌ **의사코드 (pseudocode) 금지** | 동일 이유 |
| ❌ **이미지 (그림·플롯·스크린샷) 첨부 금지** | 채점 LLM 이 이미지를 참조하지 않음 |
| ✅ **모든 결과 수치는 markdown 표로** | 표 형식이라야 자동 채점기가 정확히 파싱 |
| ⚠️ **파일 크기 100 KB 제한** | `report.md` 가 100 KB 넘으면 채점에서 잘림 |


---

## 6. 평가

채점은 다음 영역으로 나뉩니다.

| 영역 | 무엇 |
|------|------|
| 성능 | Hidden test set (300명) 으로 main.py 실행 결과 |
| 참신성 | 본인 알고리즘과 다른 학생들 알고리즘 간 similarity 비교 (낮을수록 참신) |
| 보고서 | 위 §5.1 의 4개 섹션을 종합 평가 |

세부 가중치·임계값은 비공개.

---

## 7. 마감

- **1차 제출 마감**: **2026년 6월 4일 (목) 자정**
- **2차 제출 마감**: **2026년 6월 7일 (일) 자정**

---

## 8. 질문 관련
- 모든 질문은 조교 이메일을 통해서 해주시고, 들어오는 질문들에 대해 README.md 파일을 업데이트 할 예정입니다.
- 1분반 - sanghyeok.kim@inha.edu
- 2분반 - kimjaehong@inha.edu
- jh.koo@inha.edu
