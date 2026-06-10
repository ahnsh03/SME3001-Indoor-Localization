from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def rmse_stats(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    e = np.sqrt(np.sum((y_pred - y_true) ** 2, axis=1))
    return float(np.sqrt(np.mean(e**2))), float(np.median(e)), float(np.percentile(e, 90))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = sio.loadmat(root / "data" / "InF_DH_FR1.mat", squeeze_me=True)
    x = np.asarray(raw["d_hat"], dtype=float)
    if x.shape[0] == 18:
        x = x.T  # (N, 18)
    y = np.asarray(raw["p"], dtype=float)
    if y.shape[0] == 2:
        y = y.T  # (N, 2)

    mins = x.min(axis=1, keepdims=True)
    maxs = x.max(axis=1, keepdims=True)
    means = x.mean(axis=1, keepdims=True)
    stds = x.std(axis=1, keepdims=True)
    x_feat = np.hstack([x, mins, maxs, means, stds, np.sort(x, axis=1)[:, :6]])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "knn_w_5": Pipeline(
            [("sc", StandardScaler()), ("m", KNeighborsRegressor(n_neighbors=5, weights="distance"))]
        ),
        "knn_w_9": Pipeline(
            [("sc", StandardScaler()), ("m", KNeighborsRegressor(n_neighbors=9, weights="distance"))]
        ),
        "rf_800": RandomForestRegressor(n_estimators=800, random_state=42, n_jobs=-1, min_samples_leaf=1),
        "et_800": ExtraTreesRegressor(n_estimators=800, random_state=42, n_jobs=-1, min_samples_leaf=1),
        "hgb_500": MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=6, random_state=42)
        ),
    }

    for name, model in models.items():
        pred = np.zeros_like(y)
        for tr, va in cv.split(x_feat):
            model.fit(x_feat[tr], y[tr])
            pred[va] = model.predict(x_feat[va])
        rmse, med, p90 = rmse_stats(y, pred)
        print(f"{name}: rmse={rmse:.3f} med={med:.3f} p90={p90:.3f}")


if __name__ == "__main__":
    main()
