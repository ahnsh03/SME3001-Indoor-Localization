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
    residual_knn_neighbors: int = 12  # upgraded smoothing in range 10~15
    outlier_residual_threshold_m: float = 3.0


class WifiUwbFusionLocalizerV2:
    def __init__(self, config: Config):
        self.config = config

        # Requirement-based tile coordinates
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

        # Fallback by canonical sensor order in files
        self.uwb_pos_order_m = self._tile_to_meter([(1, 4), (20, 7), (5, 15), (4, 27), (15, 14), (14, 24)])
        self.wifi_pos_order_m = self._tile_to_meter([(20, 7), (4, 27), (14, 24), (15, 14), (1, 4), (5, 15)])

        self.uwb_imputer: KNNImputer | None = None
        self.residual_knn: KNeighborsRegressor | None = None
        self.uwb_cols: List[str] = []
        self.wifi_cols: List[str] = []

    def _tile_to_meter(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        return [(x * self.config.grid_size_m, y * self.config.grid_size_m) for x, y in points]

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
        self._assert_files_exist([train_median_path, train_variance_path, val_median_path, val_variance_path])

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
                "train/validation 테이블의 센서 열 구성 불일치 (v2)."
            )

        required = list(node_cols) + self.uwb_cols + self.wifi_cols

        for name, df in [("train_median", tr_med), ("train_variance", tr_var), ("val_median", va_med), ("val_variance", va_var)]:
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
        for c in self.uwb_cols:
            out[f"med_{c}"] = imputed_df[f"med_{c}"]
        return out

    def _resolve_sensor_position_m(self, sensor_name: str, sensor_index: int, is_uwb: bool) -> Tuple[float, float]:
        if is_uwb and sensor_name in self.uwb_anchor_tiles:
            x, y = self.uwb_anchor_tiles[sensor_name]
            return x * self.config.grid_size_m, y * self.config.grid_size_m
        if (not is_uwb) and sensor_name in self.wifi_ap_tiles:
            x, y = self.wifi_ap_tiles[sensor_name]
            return x * self.config.grid_size_m, y * self.config.grid_size_m
        return (self.uwb_pos_order_m if is_uwb else self.wifi_pos_order_m)[sensor_index]

    def _sensor_hardware_key(self, col: str, idx: int, is_uwb: bool) -> str:
        if is_uwb:
            if col in self.uwb_anchor_tiles:
                return col
            return UWB_COL_CANON[idx]
        if col in self.wifi_ap_tiles:
            return col
        return WIFI_COL_CANON[idx]

    def _build_uwb_arrays(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos_list: List[Tuple[float, float]] = []
        d_list: List[float] = []
        v_list: List[float] = []
        for i, c in enumerate(self.uwb_cols):
            dr = row.get(f"med_{c}", np.nan)
            vr = row.get(f"var_{c}", np.nan)
            if not np.isfinite(dr) or not np.isfinite(vr) or float(vr) > VAR_CAP_UWB_FUSION:
                continue
            k = self._sensor_hardware_key(c, i, True)
            pos_list.append(self._resolve_sensor_position_m(c, i, is_uwb=True))
            d_list.append(float(dr) - HARDWARE_CALIB_BIAS_M[k])
            v_list.append(float(vr))
        return (
            np.asarray(pos_list, dtype=float),
            np.asarray(d_list, dtype=float),
            np.asarray(v_list, dtype=float),
        )

    def _build_wifi_arrays(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos_list: List[Tuple[float, float]] = []
        d_list: List[float] = []
        v_list: List[float] = []
        for i, c in enumerate(self.wifi_cols):
            dr = row.get(f"med_{c}", np.nan)
            vr = row.get(f"var_{c}", np.nan)
            if not np.isfinite(dr) or not np.isfinite(vr) or float(vr) > VAR_CAP_WIFI_FUSION:
                continue
            k = self._sensor_hardware_key(c, i, False)
            pos_list.append(self._resolve_sensor_position_m(c, i, is_uwb=False))
            d_list.append(float(dr) - HARDWARE_CALIB_BIAS_M[k])
            v_list.append(float(vr))
        return (
            np.asarray(pos_list, dtype=float),
            np.asarray(d_list, dtype=float),
            np.asarray(v_list, dtype=float),
        )

    def _wifi_only_initial_guess(self, row: pd.Series) -> np.ndarray:
        candidates: List[Tuple[float, Tuple[float, float]]] = []
        for i, c in enumerate(self.wifi_cols):
            d_raw = row.get(f"med_{c}", np.nan)
            v_raw = row.get(f"var_{c}", np.nan)
            if not np.isfinite(d_raw) or not np.isfinite(v_raw) or float(v_raw) > VAR_CAP_WIFI_FUSION:
                continue
            k = self._sensor_hardware_key(c, i, False)
            candidates.append(
                (
                    float(d_raw) - HARDWARE_CALIB_BIAS_M[k],
                    self._resolve_sensor_position_m(c, i, is_uwb=False),
                )
            )
        if not candidates:
            return np.array([6.0, 9.0], dtype=float)
        _, best_pos = min(candidates, key=lambda x: x[0])
        return np.array(best_pos, dtype=float)

    def _row_wifi_only_trilateration(self, row: pd.Series) -> Tuple[float, float]:
        pos, dist, var = self._build_wifi_arrays(row)
        x0 = self._wifi_only_initial_guess(row)
        xy = self._solve_weighted_trilat(pos, dist, var, x0=x0)
        return float(xy[0]), float(xy[1])

    def _smart_initial_guess(self, row: pd.Series) -> np.ndarray:
        candidates: List[Tuple[float, Tuple[float, float]]] = []
        for i, c in enumerate(self.uwb_cols):
            d = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            if not np.isfinite(d) or not np.isfinite(v) or float(v) > VAR_CAP_UWB_FUSION:
                continue
            k = self._sensor_hardware_key(c, i, True)
            candidates.append(
                (float(d) - HARDWARE_CALIB_BIAS_M[k], self._resolve_sensor_position_m(c, i, is_uwb=True))
            )
        for i, c in enumerate(self.wifi_cols):
            d = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            if not np.isfinite(d) or not np.isfinite(v) or float(v) > VAR_CAP_WIFI_FUSION:
                continue
            k = self._sensor_hardware_key(c, i, False)
            candidates.append(
                (float(d) - HARDWARE_CALIB_BIAS_M[k], self._resolve_sensor_position_m(c, i, is_uwb=False))
            )

        if not candidates:
            return np.array([6.0, 9.0], dtype=float)
        _, best_pos = min(candidates, key=lambda x: x[0])
        return np.array(best_pos, dtype=float)

    def _solve_weighted_trilat(self, pos: np.ndarray, dist: np.ndarray, var: np.ndarray, x0: np.ndarray) -> np.ndarray:
        valid = np.isfinite(dist) & np.isfinite(var)
        if valid.sum() < 3:
            valid_d = np.isfinite(dist)
            if valid_d.sum() == 0:
                return np.array([np.nan, np.nan], dtype=float)
            p = pos[valid_d]
            d = np.clip(dist[valid_d], 1e-3, None)
            w = 1.0 / d
            return (p * w[:, None]).sum(axis=0) / w.sum()

        p = pos[valid]
        d = dist[valid]
        v = var[valid]
        w = 1.0 / (v + self.config.epsilon)

        def residuals(xy: np.ndarray) -> np.ndarray:
            pred = np.sqrt((xy[0] - p[:, 0]) ** 2 + (xy[1] - p[:, 1]) ** 2)
            return np.sqrt(w) * (pred - d)

        res = least_squares(
            residuals,
            x0=x0,
            loss=self.config.trilateration_loss,
            f_scale=self.config.huber_f_scale,
            method="trf",
        )
        return np.array(res.x, dtype=float)

    def _row_step_ab(self, row: pd.Series) -> Tuple[float, float, float, float]:
        pos, dist, var = self._build_uwb_arrays(row)

        # Step A: smart initial guess + robust weighted trilateration
        x0 = self._smart_initial_guess(row)
        step_a = self._solve_weighted_trilat(pos, dist, var, x0=x0)

        # Step B: IRLS-like outlier pruning and refit
        valid = np.isfinite(dist) & np.isfinite(var)
        if valid.sum() < 3 or not np.all(np.isfinite(step_a)):
            return float(step_a[0]), float(step_a[1]), float(step_a[0]), float(step_a[1])

        residual_geom = np.abs(np.sqrt((step_a[0] - pos[:, 0]) ** 2 + (step_a[1] - pos[:, 1]) ** 2) - dist)
        keep_mask = valid & (residual_geom <= self.config.outlier_residual_threshold_m)

        # Ensure enough anchors remain for refit
        if keep_mask.sum() < 3:
            step_b = step_a
        else:
            pruned_var = var.copy()
            pruned_var[~keep_mask] = np.inf  # weight -> 0
            step_b = self._solve_weighted_trilat(pos, dist, pruned_var, x0=step_a)

        return float(step_a[0]), float(step_a[1]), float(step_b[0]), float(step_b[1])

    def run_step_ab_trilateration(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        ab = out.apply(self._row_step_ab, axis=1, result_type="expand")
        ab.columns = ["StepA_X", "StepA_Y", "StepB_X", "StepB_Y"]
        return pd.concat([out, ab], axis=1)

    def _residual_feature_cols(self) -> List[str]:
        med_cols = [f"med_{c}" for c in (self.uwb_cols + self.wifi_cols)]
        var_cols = [f"var_{c}" for c in (self.uwb_cols + self.wifi_cols)]
        return med_cols + var_cols

    def fit_residual_knn(self, train_df: pd.DataFrame) -> None:
        out = train_df.copy()
        out["dx"] = out["True_X"] - out["StepB_X"]
        out["dy"] = out["True_Y"] - out["StepB_Y"]

        feat_cols = self._residual_feature_cols()
        valid = out[feat_cols + ["dx", "dy"]].dropna()

        self.residual_knn = KNeighborsRegressor(
            n_neighbors=self.config.residual_knn_neighbors,
            weights="distance",
            metric="minkowski",
            p=2,
        )
        self.residual_knn.fit(valid[feat_cols], valid[["dx", "dy"]])

    def apply_knn_correction_step_c(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.residual_knn is None:
            raise RuntimeError("Residual KNN is not fitted.")
        out = df.copy()
        feat_cols = self._residual_feature_cols()
        feat = out[feat_cols].copy().fillna(out[feat_cols].median(numeric_only=True))
        pred_delta = self.residual_knn.predict(feat)
        out["Pred_dx"] = pred_delta[:, 0]
        out["Pred_dy"] = pred_delta[:, 1]
        out["Final_X"] = out["StepB_X"] + out["Pred_dx"]
        out["Final_Y"] = out["StepB_Y"] + out["Pred_dy"]
        return out

    @staticmethod
    def _point_error(true_x: np.ndarray, true_y: np.ndarray, pred_x: np.ndarray, pred_y: np.ndarray) -> np.ndarray:
        return np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)

    def evaluate_steps(self, df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray | float]]:
        valid = df.dropna(subset=["True_X", "True_Y", "StepA_X", "StepA_Y", "StepB_X", "StepB_Y", "Final_X", "Final_Y"])

        err_a = self._point_error(valid["True_X"].values, valid["True_Y"].values, valid["StepA_X"].values, valid["StepA_Y"].values)
        err_b = self._point_error(valid["True_X"].values, valid["True_Y"].values, valid["StepB_X"].values, valid["StepB_Y"].values)
        err_c = self._point_error(valid["True_X"].values, valid["True_Y"].values, valid["Final_X"].values, valid["Final_Y"].values)

        def rmse_mae(err: np.ndarray) -> Tuple[float, float]:
            rmse = float(np.sqrt(mean_squared_error(np.zeros_like(err), err)))
            mae = float(mean_absolute_error(np.zeros_like(err), err))
            return rmse, mae

        rmse_a, mae_a = rmse_mae(err_a)
        rmse_b, mae_b = rmse_mae(err_b)
        rmse_c, mae_c = rmse_mae(err_c)

        return {
            "A": {"RMSE_m": rmse_a, "MAE_m": mae_a, "Errors": err_a},
            "B": {"RMSE_m": rmse_b, "MAE_m": mae_b, "Errors": err_b},
            "C": {"RMSE_m": rmse_c, "MAE_m": mae_c, "Errors": err_c},
        }

    @staticmethod
    def plot_cdf_comparison(err_a: np.ndarray, err_b: np.ndarray, err_c: np.ndarray, out_path: str) -> None:
        plt.figure(figsize=(8, 5))
        for errs, label, color in [
            (err_a, "Step A: Smart Guess Trilateration", "tab:blue"),
            (err_b, "Step B: IRLS Trilateration", "tab:orange"),
            (err_c, "Step C: K-NN Error Correction", "tab:green"),
        ]:
            se = np.sort(errs)
            cdf = np.arange(1, len(se) + 1) / len(se)
            plt.plot(se, cdf, linewidth=2, label=label, color=color)

        plt.xlabel("Position Error (m)")
        plt.ylabel("Cumulative Probability")
        plt.title("Validation Error CDF Comparison (Step A/B/C)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()


def main() -> None:
    cfg = Config(
        grid_size_m=0.6,
        epsilon=1e-4,
        trilateration_loss="huber",
        huber_f_scale=1.0,
        uwb_imputer_neighbors=5,
        residual_knn_neighbors=12,
        outlier_residual_threshold_m=3.0,
    )
    localizer = WifiUwbFusionLocalizerV2(cfg)

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
    print(f"V2: 학습 {len(train)}행 ({train_med.name}) · 검증 {len(val)}행 ({val_med.name})")

    # Phase 1
    localizer.fit_uwb_imputer(train)
    train_imp = localizer.apply_uwb_imputation(train)
    val_imp = localizer.apply_uwb_imputation(val)

    # Phase 2 (Step A & Step B)
    train_ab = localizer.run_step_ab_trilateration(train_imp)
    val_ab = localizer.run_step_ab_trilateration(val_imp)

    # Phase 3 (Step C)
    localizer.fit_residual_knn(train_ab)
    val_final = localizer.apply_knn_correction_step_c(val_ab)

    # Step-by-step evaluation
    step_metrics = localizer.evaluate_steps(val_final)

    wf_x = np.full(len(val_imp), np.nan)
    wf_y = np.full(len(val_imp), np.nan)
    for i, (_, row) in enumerate(val_imp.iterrows()):
        wx, wy = localizer._row_wifi_only_trilateration(row)
        wf_x[i], wf_y[i] = wx, wy
    err_wifi = WifiUwbFusionLocalizerV2._point_error(
        val_imp["True_X"].values, val_imp["True_Y"].values, wf_x, wf_y
    )
    fin_w = np.isfinite(err_wifi)
    ewf = err_wifi[fin_w]
    rmse_wifi = (
        float(np.sqrt(mean_squared_error(np.zeros_like(ewf), ewf))) if ewf.size else float("nan")
    )
    mae_wifi = float(mean_absolute_error(np.zeros_like(ewf), ewf)) if ewf.size else float("nan")
    print("=== Validation (Wi-Fi only 삼변, UWB 미사용 · V2 알고리즘) ===")
    print(f"Wi-Fi-only RMSE: {rmse_wifi:.4f} m | MAE: {mae_wifi:.4f} m")

    print("=== Step-by-Step Validation Performance ===")
    print(f"Step A RMSE: {step_metrics['A']['RMSE_m']:.4f} m | MAE: {step_metrics['A']['MAE_m']:.4f} m")
    print(f"Step B RMSE: {step_metrics['B']['RMSE_m']:.4f} m | MAE: {step_metrics['B']['MAE_m']:.4f} m")
    print(f"Step C RMSE: {step_metrics['C']['RMSE_m']:.4f} m | MAE: {step_metrics['C']['MAE_m']:.4f} m")

    # Save prediction CSV with all requested coordinate stages
    prediction_cols = [
        "Node_x",
        "Node_y",
        "True_X",
        "True_Y",
        "StepA_X",
        "StepA_Y",
        "StepB_X",
        "StepB_Y",
        "Final_X",
        "Final_Y",
    ]
    pred_path = os.path.join(output_dir, "v2_validation_predictions.csv")
    val_final[prediction_cols].to_csv(pred_path, index=False)

    # Save step RMSE report CSV
    report_df = pd.DataFrame(
        [
            {"Step": "A", "Method": "Smart Guess Trilateration", "RMSE_m": step_metrics["A"]["RMSE_m"], "MAE_m": step_metrics["A"]["MAE_m"]},
            {"Step": "B", "Method": "IRLS Trilateration", "RMSE_m": step_metrics["B"]["RMSE_m"], "MAE_m": step_metrics["B"]["MAE_m"]},
            {"Step": "C", "Method": "K-NN Error Correction", "RMSE_m": step_metrics["C"]["RMSE_m"], "MAE_m": step_metrics["C"]["MAE_m"]},
        ]
    )
    report_path = os.path.join(output_dir, "v2_step_rmse_report.csv")
    report_df.to_csv(report_path, index=False)

    # Save overlapped CDF comparison
    cdf_path = os.path.join(output_dir, "v2_validation_error_cdf_steps_abc.png")
    localizer.plot_cdf_comparison(
        step_metrics["A"]["Errors"],
        step_metrics["B"]["Errors"],
        step_metrics["C"]["Errors"],
        out_path=cdf_path,
    )

    print(f"Saved: {pred_path}")
    print(f"Saved: {report_path}")
    print(f"Saved: {cdf_path}")


if __name__ == "__main__":
    main()
