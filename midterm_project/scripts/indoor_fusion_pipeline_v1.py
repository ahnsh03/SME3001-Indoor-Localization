import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

from fusion_realtime_sanitize import (
    HARDWARE_CALIB_BIAS_M,
    VAR_CAP_UWB_FUSION,
    VAR_CAP_WIFI_FUSION,
    infer_sensor_columns_ordered,
    resolve_train_kgh_corrected_paths,
    resolve_validation_paths,
    standardize_loaded_median_variance,
)

UWB_COL_CANON = ["110394ab", "e63ce2f", "8e610981", "d10485af", "d1044709", "4e610206"]
WIFI_COL_CANON = ["SW_11", "SW_first_team", "볼링공", "SW_4", "SW_6", "SW_5"]


@dataclass
class Config:
    grid_size_m: float = 0.6
    epsilon: float = 1e-4
    trilateration_loss: str = "huber"  # 'huber' or 'soft_l1'
    huber_f_scale: float = 1.0
    uwb_imputer_neighbors: int = 5
    residual_knn_neighbors: int = 7


class WifiUwbFusionLocalizer:
    def __init__(self, config: Config):
        self.config = config

        # Tile coordinates from requirement
        self.uwb_anchor_tiles: Dict[str, Tuple[float, float]] = {
            "110394ab": (1, 4),
            "e63ce2f": (20, 7),
            "8e610981": (5, 15),
            "d10485af": (4, 27),
            "d1044709": (15, 14),
            "4e610206": (14, 24),
        }
        self.wifi_ap_tiles: Dict[str, Tuple[float, float]] = {
            "SW_11": (1, 4),
            "SW_first_team": (20, 7),
            "볼링공": (5, 15),
            "SW_4": (4, 27),
            "SW_6": (15, 14),
            "SW_5": (14, 24),
        }

        self.uwb_imputer: KNNImputer | None = None
        self.residual_knn: KNeighborsRegressor | None = None

        self.uwb_cols: List[str] = []
        self.wifi_cols: List[str] = []

    def _assert_files_exist(self, file_paths: List[str]) -> None:
        missing = [p for p in file_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing required files: {missing}")

    def _coerce_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    def load_datasets(
        self,
        train_median_path: str,
        train_variance_path: str,
        val_median_path: str,
        val_variance_path: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._assert_files_exist([
            train_median_path,
            train_variance_path,
            val_median_path,
            val_variance_path,
        ])

        tm, tv = Path(train_median_path), Path(train_variance_path)
        vm, vv = Path(val_median_path), Path(val_variance_path)
        tr_med, tr_var, node_cols, self.uwb_cols, self.wifi_cols = standardize_loaded_median_variance(
            tm, tv, self._coerce_numeric, infer_sensor_columns_ordered
        )
        va_med, va_var, _nc2, uwb_v2, wf_v2 = standardize_loaded_median_variance(
            vm, vv, self._coerce_numeric, infer_sensor_columns_ordered
        )
        if self.uwb_cols != uwb_v2 or self.wifi_cols != wf_v2:
            raise ValueError(
                "train/validation 테이블의 센서 열 구성 불일치: "
                f"train {[self.uwb_cols, self.wifi_cols]} vs val {[uwb_v2, wf_v2]}"
            )

        required = list(node_cols) + self.uwb_cols + self.wifi_cols
        for name, df in [
            ("train_median", tr_med),
            ("train_variance", tr_var),
            ("val_median", va_med),
            ("val_variance", va_var),
        ]:
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"{name} missing columns: {missing}")

        train = pd.DataFrame()
        val = pd.DataFrame()

        for target, med, var in [(train, tr_med, tr_var), (val, va_med, va_var)]:
            target["Node_x"] = med[node_cols[0]]
            target["Node_y"] = med[node_cols[1]]
            target["True_X"] = target["Node_x"] * self.config.grid_size_m
            target["True_Y"] = target["Node_y"] * self.config.grid_size_m

            for c in self.uwb_cols:
                target[f"med_{c}"] = med[c]
                target[f"var_{c}"] = var[c]

            for c in self.wifi_cols:
                target[f"med_{c}"] = med[c]
                target[f"var_{c}"] = var[c]

        return train, val

    def fit_uwb_imputer(self, train_df: pd.DataFrame) -> None:
        feature_cols = [f"med_{c}" for c in self.wifi_cols] + [f"med_{c}" for c in self.uwb_cols]
        self.uwb_imputer = KNNImputer(n_neighbors=self.config.uwb_imputer_neighbors, weights="distance")
        self.uwb_imputer.fit(train_df[feature_cols])

    def apply_uwb_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.uwb_imputer is None:
            raise RuntimeError("UWB imputer is not fitted.")

        out = df.copy()
        feature_cols = [f"med_{c}" for c in self.wifi_cols] + [f"med_{c}" for c in self.uwb_cols]
        imputed = self.uwb_imputer.transform(out[feature_cols])
        imputed_df = pd.DataFrame(imputed, columns=feature_cols, index=out.index)

        # Replace only UWB medians with imputed values; Wi-Fi remains observed reference features
        for c in self.uwb_cols:
            out[f"med_{c}"] = imputed_df[f"med_{c}"]

        return out

    def _anchor_positions_m(self) -> Dict[str, Tuple[float, float]]:
        return {k: (v[0] * self.config.grid_size_m, v[1] * self.config.grid_size_m) for k, v in self.uwb_anchor_tiles.items()}

    def _sensor_hardware_key(self, col: str, idx: int, is_uwb: bool) -> str:
        if is_uwb:
            if col in self.uwb_anchor_tiles:
                return col
            return UWB_COL_CANON[idx]
        if col in self.wifi_ap_tiles:
            return col
        return WIFI_COL_CANON[idx]

    def _row_trilateration(self, row: pd.Series) -> Tuple[float, float]:
        anchors = self._anchor_positions_m()
        pos_rows: List[np.ndarray] = []
        d_cal: List[float] = []
        weights: List[float] = []

        for i, c in enumerate(self.uwb_cols):
            dr = row.get(f"med_{c}", np.nan)
            vr = row.get(f"var_{c}", np.nan)
            if not np.isfinite(dr) or not np.isfinite(vr) or float(vr) > VAR_CAP_UWB_FUSION:
                continue
            if c not in anchors:
                continue
            k = self._sensor_hardware_key(c, i, True)
            pos_rows.append(np.array(anchors[c], dtype=float))
            d_cal.append(float(dr) - HARDWARE_CALIB_BIAS_M[k])
            weights.append(1.0 / (float(vr) + self.config.epsilon))

        if len(d_cal) < 3:
            valid_d = np.isfinite(np.array([row.get(f"med_{c}", np.nan) for c in self.uwb_cols], dtype=float))
            if valid_d.sum() == 0:
                return np.nan, np.nan
            ds = np.array([row[f"med_{c}"] for c in self.uwb_cols], dtype=float)
            pos = np.array([anchors[c] for c in self.uwb_cols], dtype=float)[valid_d]
            d = np.clip(ds[valid_d], 1e-3, None)
            wgt = 1.0 / d
            est = (pos * wgt[:, None]).sum(axis=0) / wgt.sum()
            return float(est[0]), float(est[1])

        pos = np.stack(pos_rows, axis=0)
        d = np.asarray(d_cal, dtype=float)
        ww = np.asarray(weights, dtype=float)

        def residuals(p: np.ndarray) -> np.ndarray:
            pred = np.sqrt((p[0] - pos[:, 0]) ** 2 + (p[1] - pos[:, 1]) ** 2)
            return np.sqrt(ww) * (pred - d)

        x0 = np.average(pos, axis=0, weights=ww)
        result = least_squares(
            residuals,
            x0=x0,
            loss=self.config.trilateration_loss,
            f_scale=self.config.huber_f_scale,
            method="trf",
        )

        return float(result.x[0]), float(result.x[1])

    def _resolve_wifi_anchor_m(self, sensor_name: str, sensor_index: int) -> Tuple[float, float]:
        g = self.config.grid_size_m
        if sensor_name in self.wifi_ap_tiles:
            tx, ty = self.wifi_ap_tiles[sensor_name]
            return tx * g, ty * g
        order_tiles = [(20, 7), (4, 27), (14, 24), (15, 14), (1, 4), (5, 15)]
        tx, ty = order_tiles[sensor_index] if sensor_index < len(order_tiles) else (10.0, 9.0)
        return tx * g, ty * g

    def _row_wifi_trilateration(self, row: pd.Series) -> Tuple[float, float]:
        """UWB 미사용. PHASE1 거리 오프셋·분산 상한 적용 후 가중 Huber 삼변."""
        pos_rows: List[np.ndarray] = []
        d_cal: List[float] = []
        weights: List[float] = []
        pos_all = np.array(
            [self._resolve_wifi_anchor_m(c, i) for i, c in enumerate(self.wifi_cols)],
            dtype=float,
        )

        for i, c in enumerate(self.wifi_cols):
            dr = row.get(f"med_{c}", np.nan)
            vr = row.get(f"var_{c}", np.nan)
            if not np.isfinite(dr) or not np.isfinite(vr) or float(vr) > VAR_CAP_WIFI_FUSION:
                continue
            k = self._sensor_hardware_key(c, i, False)
            pos_rows.append(pos_all[i].copy())
            d_cal.append(float(dr) - HARDWARE_CALIB_BIAS_M[k])
            weights.append(1.0 / (float(vr) + self.config.epsilon))

        if len(d_cal) < 3:
            ds = np.array([row.get(f"med_{c}", np.nan) for c in self.wifi_cols], dtype=float)
            valid_d = np.isfinite(ds)
            if valid_d.sum() == 0:
                return np.nan, np.nan
            pos = pos_all[valid_d]
            d = np.clip(ds[valid_d], 1e-3, None)
            w = 1.0 / d
            est = (pos * w[:, None]).sum(axis=0) / w.sum()
            return float(est[0]), float(est[1])

        pos = np.stack(pos_rows, axis=0)
        d = np.asarray(d_cal, dtype=float)
        ww = np.asarray(weights, dtype=float)

        def residuals(p: np.ndarray) -> np.ndarray:
            pred = np.sqrt((p[0] - pos[:, 0]) ** 2 + (p[1] - pos[:, 1]) ** 2)
            return np.sqrt(ww) * (pred - d)

        x0 = np.average(pos, axis=0, weights=ww)
        result = least_squares(
            residuals,
            x0=x0,
            loss=self.config.trilateration_loss,
            f_scale=self.config.huber_f_scale,
            method="trf",
        )
        return float(result.x[0]), float(result.x[1])

    def run_base_trilateration(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        base_xy = out.apply(self._row_trilateration, axis=1, result_type="expand")
        base_xy.columns = ["Base_X", "Base_Y"]
        out = pd.concat([out, base_xy], axis=1)
        return out

    def _residual_feature_cols(self) -> List[str]:
        med_cols = [f"med_{c}" for c in (self.uwb_cols + self.wifi_cols)]
        var_cols = [f"var_{c}" for c in (self.uwb_cols + self.wifi_cols)]
        return med_cols + var_cols

    def fit_residual_knn(self, train_df: pd.DataFrame) -> None:
        out = train_df.copy()
        out["dx"] = out["True_X"] - out["Base_X"]
        out["dy"] = out["True_Y"] - out["Base_Y"]

        feat_cols = self._residual_feature_cols()
        valid = out[feat_cols + ["dx", "dy"]].dropna()

        self.residual_knn = KNeighborsRegressor(
            n_neighbors=self.config.residual_knn_neighbors,
            weights="distance",
            metric="minkowski",
            p=2,
        )
        self.residual_knn.fit(valid[feat_cols], valid[["dx", "dy"]])

    def apply_residual_correction(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.residual_knn is None:
            raise RuntimeError("Residual KNN is not fitted.")

        out = df.copy()
        feat_cols = self._residual_feature_cols()

        feat = out[feat_cols].copy()
        feat = feat.fillna(feat.median(numeric_only=True))

        pred_delta = self.residual_knn.predict(feat)
        out["Pred_dx"] = pred_delta[:, 0]
        out["Pred_dy"] = pred_delta[:, 1]
        out["Final_X"] = out["Base_X"] + out["Pred_dx"]
        out["Final_Y"] = out["Base_Y"] + out["Pred_dy"]

        return out

    @staticmethod
    def evaluate(df: pd.DataFrame) -> Dict[str, float]:
        valid = df[["True_X", "True_Y", "Final_X", "Final_Y"]].dropna()
        true_xy = valid[["True_X", "True_Y"]].values
        pred_xy = valid[["Final_X", "Final_Y"]].values

        errors = np.linalg.norm(pred_xy - true_xy, axis=1)
        rmse = float(np.sqrt(mean_squared_error(np.zeros_like(errors), errors)))
        mae = float(mean_absolute_error(np.zeros_like(errors), errors))

        return {"RMSE_m": rmse, "MAE_m": mae, "Errors": errors}

    @staticmethod
    def plot_cdf(errors: np.ndarray, out_path: str = "v1_validation_error_cdf.png") -> None:
        sorted_err = np.sort(errors)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)

        plt.figure(figsize=(8, 5))
        plt.plot(sorted_err, cdf, linewidth=2)
        plt.xlabel("Position Error (m)")
        plt.ylabel("Cumulative Probability")
        plt.title("Validation Error CDF")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    @staticmethod
    def plot_error_vector_map(df: pd.DataFrame, out_path: str = "v1_validation_error_vector_map.png") -> None:
        valid = df[["True_X", "True_Y", "Final_X", "Final_Y"]].dropna()

        tx = valid["True_X"].values
        ty = valid["True_Y"].values
        px = valid["Final_X"].values
        py = valid["Final_Y"].values

        plt.figure(figsize=(10, 6))
        plt.scatter(tx, ty, c="blue", s=25, label="True Position", alpha=0.8)
        plt.scatter(px, py, c="red", s=25, label="Estimated Position", alpha=0.8)

        # Error vectors: True -> Estimated
        plt.quiver(
            tx,
            ty,
            px - tx,
            py - ty,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="gray",
            alpha=0.6,
            width=0.0025,
        )

        plt.xlim(0, 12)
        plt.ylim(0, 18)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("2D Error Vector Map (Validation)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()


def main() -> int:
    cfg = Config(
        grid_size_m=0.6,
        epsilon=1e-4,
        trilateration_loss="huber",
        huber_f_scale=1.0,
        uwb_imputer_neighbors=5,
        residual_knn_neighbors=7,
    )

    localizer = WifiUwbFusionLocalizer(cfg)

    project_root = Path(__file__).resolve().parent.parent
    train_dir = project_root / "data" / "train"
    val_dir = project_root / "data" / "validation"
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_med, train_var = resolve_train_kgh_corrected_paths(train_dir)
    val_med, val_var = resolve_validation_paths(val_dir)
    train, val = localizer.load_datasets(
        train_median_path=str(train_med),
        train_variance_path=str(train_var),
        val_median_path=str(val_med),
        val_variance_path=str(val_var),
    )
    print(f"V1: 학습 {len(train)}행 ({train_med.name}) · 검증 {len(val)}행 ({val_med.name})")

    # Phase 1: KNN-based UWB missing imputation using Wi-Fi context
    localizer.fit_uwb_imputer(train)
    train_imp = localizer.apply_uwb_imputation(train)
    val_imp = localizer.apply_uwb_imputation(val)

    wf_x = np.full(len(val_imp), np.nan, dtype=float)
    wf_y = np.full(len(val_imp), np.nan, dtype=float)
    for i, (_, row) in enumerate(val_imp.iterrows()):
        wx, wy = localizer._row_wifi_trilateration(row)
        wf_x[i], wf_y[i] = wx, wy
    err_wifi = np.hypot(wf_x - val_imp["True_X"].values, wf_y - val_imp["True_Y"].values)
    fin = np.isfinite(err_wifi)
    if fin.any():
        e = err_wifi[fin]
        rmse_wifi = float(np.sqrt(mean_squared_error(np.zeros_like(e), e)))
        mae_wifi = float(mean_absolute_error(np.zeros_like(e), e))
    else:
        rmse_wifi = float("nan")
        mae_wifi = float("nan")
    print("=== Validation (Wi-Fi only 삼변, UWB 미사용 · V1 알고리즘) ===")
    print(f"Wi-Fi-only RMSE: {rmse_wifi:.4f} m | MAE: {mae_wifi:.4f} m")

    # Phase 2: Variance-weighted robust trilateration
    train_base = localizer.run_base_trilateration(train_imp)
    val_base = localizer.run_base_trilateration(val_imp)

    # Phase 3: KNN residual error mapping and correction
    localizer.fit_residual_knn(train_base)
    val_final = localizer.apply_residual_correction(val_base)

    # Phase 4: Evaluation and visualization
    metrics = localizer.evaluate(val_final)
    print("=== Validation Metrics ===")
    print(f"RMSE: {metrics['RMSE_m']:.4f} m")
    print(f"MAE : {metrics['MAE_m']:.4f} m")

    cdf_path = os.path.join(output_dir, "v1_validation_error_cdf.png")
    map_path = os.path.join(output_dir, "v1_validation_error_vector_map.png")
    pred_path = os.path.join(output_dir, "v1_validation_predictions.csv")

    localizer.plot_cdf(metrics["Errors"], out_path=cdf_path)
    localizer.plot_error_vector_map(val_final, out_path=map_path)

    val_final.to_csv(pred_path, index=False)
    return 0


if __name__ == "__main__":
    from pathlib import Path

    from script_run_io import cli_entrypoint

    cli_entrypoint(Path(__file__), main, output_artifact_include_prefixes=("v1_",))
