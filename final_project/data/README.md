# 기말 데이터

| 파일 | 설명 |
|------|------|
| `InF_DH_FR1.mat` | 학습용 합성 RTT (`p`, `d_hat`, `BS_positions` 등). 채점 시 `DH_FR1.mat`과 동일 내용. |

로더 예:

```python
import scipy.io as sio
from pathlib import Path

mat_path = Path(__file__).resolve().parents[1] / "data" / "InF_DH_FR1.mat"
data = sio.loadmat(mat_path, squeeze_me=True)
```
