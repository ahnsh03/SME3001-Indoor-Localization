"""V3 Wi‑Fi/UWB 융합 파이프라인.

데이터 무결성: UWB 결측용 KNN imputer와 Phase-3 잔차 KNN은 **train에만** fit한다.
검증 세트의 정답(True_X/Y)은 **평가·시각화·CSV 저장**에만 쓰이며,
검증 RMSE로 하이퍼파라미터를 고르는 그리드 탐색은 없다.
"""

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
    max_physical_distance_m: float = 22.0
    epsilon: float = 1e-4
    trilateration_loss: str = "huber"
    huber_f_scale: float = 1.0
    uwb_bias: float = 0.5
    wifi_bias: float = 2.5
    uwb_imputer_neighbors: int = 5
    phase3_knn_neighbors: int = 15


class WifiUwbFusionLocalizerV3:
    def __init__(self, config: Config):
        self.config = config
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
        self.uwb_pos_order_m = self._tile_to_meter([(1, 4), (20, 7), (5, 15), (4, 27), (15, 14), (14, 24)])
        self.wifi_pos_order_m = self._tile_to_meter([(20, 7), (4, 27), (14, 24), (15, 14), (1, 4), (5, 15)])

        self.uwb_cols: List[str] = []
        self.wifi_cols: List[str] = []
        self.uwb_imputer: KNNImputer | None = None
        self.residual_knn: KNeighborsRegressor | None = None

    def _sensor_hardware_key(self, col: str, idx: int, is_uwb: bool) -> str:
        if is_uwb:
            if col in self.uwb_anchor_tiles:
                return col
            return UWB_COL_CANON[idx]
        if col in self.wifi_ap_tiles:
            return col
        return WIFI_COL_CANON[idx]

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

    def _apply_hard_filter(self, df: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
        out = df.copy()
        for c in sensor_cols:
            out[c] = out[c].where(out[c] <= self.config.max_physical_distance_m, np.nan)
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
            raise ValueError("train/validation 테이블의 센서 열 구성 불일치 (v3).")

        sensor_cols = self.uwb_cols + self.wifi_cols
        tr_med = self._apply_hard_filter(tr_med, sensor_cols)
        va_med = self._apply_hard_filter(va_med, sensor_cols)

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
                target[f"orig_nan_{c}"] = med[c].isna().astype(int)
            for c in self.wifi_cols:
                target[f"med_{c}"] = med[c]
                target[f"var_{c}"] = var[c]

        return train, val

    def fit_uwb_imputer(self, train_df: pd.DataFrame) -> None:
        feat_cols = [f"med_{c}" for c in self.wifi_cols] + [f"med_{c}" for c in self.uwb_cols]
        self.uwb_imputer = KNNImputer(n_neighbors=self.config.uwb_imputer_neighbors, weights="distance")
        self.uwb_imputer.fit(train_df[feat_cols])

    def add_imputed_uwb_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.uwb_imputer is None:
            raise RuntimeError("UWB imputer is not fitted.")
        out = df.copy()
        feat_cols = [f"med_{c}" for c in self.wifi_cols] + [f"med_{c}" for c in self.uwb_cols]
        imputed = self.uwb_imputer.transform(out[feat_cols])
        imp_df = pd.DataFrame(imputed, columns=feat_cols, index=out.index)
        for c in self.uwb_cols:
            out[f"med_imp_{c}"] = imp_df[f"med_{c}"]
        return out

    def _resolve_sensor_position_m(self, sensor_name: str, sensor_index: int, is_uwb: bool) -> Tuple[float, float]:
        if is_uwb and sensor_name in self.uwb_anchor_tiles:
            x, y = self.uwb_anchor_tiles[sensor_name]
            return x * self.config.grid_size_m, y * self.config.grid_size_m
        if (not is_uwb) and sensor_name in self.wifi_ap_tiles:
            x, y = self.wifi_ap_tiles[sensor_name]
            return x * self.config.grid_size_m, y * self.config.grid_size_m
        return (self.uwb_pos_order_m if is_uwb else self.wifi_pos_order_m)[sensor_index]

    def _build_phase2_arrays(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos_list: List[Tuple[float, float]] = []
        dist_list: List[float] = []
        weight_list: List[float] = []

        # UWB: PHASE1 거리 오프셋 + 분산 fusion cap (sensor_spatial_profiler 정렬)
        for i, c in enumerate(self.uwb_cols):
            d = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            was_nan = row.get(f"orig_nan_{c}", 1) == 1
            if was_nan or (not np.isfinite(d)) or (not np.isfinite(v)):
                continue
            if float(v) > VAR_CAP_UWB_FUSION:
                continue
            bk = self._sensor_hardware_key(c, i, True)
            pos_list.append(self._resolve_sensor_position_m(c, i, is_uwb=True))
            dist_list.append(float(d) - HARDWARE_CALIB_BIAS_M[bk])
            weight_list.append(1.0 / (float(v) + self.config.uwb_bias + self.config.epsilon))

        # Wi‑Fi
        for i, c in enumerate(self.wifi_cols):
            d = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            if (not np.isfinite(d)) or (not np.isfinite(v)):
                continue
            if float(v) > VAR_CAP_WIFI_FUSION:
                continue
            bk = self._sensor_hardware_key(c, i, False)
            pos_list.append(self._resolve_sensor_position_m(c, i, is_uwb=False))
            dist_list.append(float(d) - HARDWARE_CALIB_BIAS_M[bk])
            weight_list.append(1.0 / (float(v) + self.config.wifi_bias + self.config.epsilon))

        return np.array(pos_list, dtype=float), np.array(dist_list, dtype=float), np.array(weight_list, dtype=float)

    def _build_wifi_only_arrays(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos_list: List[Tuple[float, float]] = []
        dist_list: List[float] = []
        weight_list: List[float] = []
        for i, c in enumerate(self.wifi_cols):
            d = row.get(f"med_{c}", np.nan)
            v = row.get(f"var_{c}", np.nan)
            if (not np.isfinite(d)) or (not np.isfinite(v)):
                continue
            if float(v) > VAR_CAP_WIFI_FUSION:
                continue
            bk = self._sensor_hardware_key(c, i, False)
            pos_list.append(self._resolve_sensor_position_m(c, i, is_uwb=False))
            dist_list.append(float(d) - HARDWARE_CALIB_BIAS_M[bk])
            weight_list.append(1.0 / (float(v) + self.config.wifi_bias + self.config.epsilon))
        return np.array(pos_list, dtype=float), np.array(dist_list, dtype=float), np.array(weight_list, dtype=float)

    def run_wifi_only_trilateration(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        xs: List[float] = []
        ys: List[float] = []
        for _, row in out.iterrows():
            pos, dist, weights = self._build_wifi_only_arrays(row)
            xy = self._solve_trilat(pos, dist, weights)
            xs.append(float(xy[0]))
            ys.append(float(xy[1]))
        out["WifiOnly_X"] = xs
        out["WifiOnly_Y"] = ys
        return out

    def _smart_initial_guess(self, pos: np.ndarray, dist: np.ndarray) -> np.ndarray:
        if len(dist) == 0:
            return np.array([6.0, 9.0], dtype=float)
        idx = int(np.argmin(dist))
        return np.array(pos[idx], dtype=float)

    def _solve_trilat(self, pos: np.ndarray, dist: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if len(dist) < 3:
            if len(dist) == 0:
                return np.array([np.nan, np.nan], dtype=float)
            d = np.clip(dist, 1e-3, None)
            w = 1.0 / d
            return (pos * w[:, None]).sum(axis=0) / w.sum()

        x0 = self._smart_initial_guess(pos, dist)

        def residuals(xy: np.ndarray) -> np.ndarray:
            pred = np.sqrt((xy[0] - pos[:, 0]) ** 2 + (xy[1] - pos[:, 1]) ** 2)
            return np.sqrt(weights) * (pred - dist)

        res = least_squares(
            residuals,
            x0=x0,
            loss=self.config.trilateration_loss,
            f_scale=self.config.huber_f_scale,
            method="trf",
        )
        return np.array(res.x, dtype=float)

    def run_step_a_pure_math(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        xs: List[float] = []
        ys: List[float] = []
        for _, row in out.iterrows():
            pos, dist, weights = self._build_phase2_arrays(row)
            xy = self._solve_trilat(pos, dist, weights)
            xs.append(float(xy[0]))
            ys.append(float(xy[1]))
        out["StepA_X"] = xs
        out["StepA_Y"] = ys
        return out

    def fit_phase3_knn(self, train_df: pd.DataFrame) -> None:
        out = train_df.copy()
        out["dx"] = out["True_X"] - out["StepA_X"]
        out["dy"] = out["True_Y"] - out["StepA_Y"]

        feat_cols = ["StepA_X", "StepA_Y"]
        valid = out[feat_cols + ["dx", "dy"]].dropna()
        self.residual_knn = KNeighborsRegressor(
            n_neighbors=self.config.phase3_knn_neighbors,
            weights="distance",
            metric="minkowski",
            p=2,
        )
        self.residual_knn.fit(valid[feat_cols], valid[["dx", "dy"]])

    def apply_step_b_final_hybrid(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.residual_knn is None:
            raise RuntimeError("Phase3 KNN is not fitted.")
        out = df.copy()
        feat = out[["StepA_X", "StepA_Y"]].copy()
        feat = feat.fillna(feat.median(numeric_only=True))
        pred = self.residual_knn.predict(feat)
        out["Pred_dx"] = pred[:, 0]
        out["Pred_dy"] = pred[:, 1]
        out["Final_X"] = out["StepA_X"] + out["Pred_dx"]
        out["Final_Y"] = out["StepA_Y"] + out["Pred_dy"]
        return out

    @staticmethod
    def point_error(true_x: np.ndarray, true_y: np.ndarray, pred_x: np.ndarray, pred_y: np.ndarray) -> np.ndarray:
        return np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)

    def evaluate_step_a_b(self, df: pd.DataFrame) -> Dict[str, Dict[str, float | np.ndarray]]:
        valid = df.dropna(subset=["True_X", "True_Y", "StepA_X", "StepA_Y", "Final_X", "Final_Y"])
        err_a = self.point_error(valid["True_X"].values, valid["True_Y"].values, valid["StepA_X"].values, valid["StepA_Y"].values)
        err_b = self.point_error(valid["True_X"].values, valid["True_Y"].values, valid["Final_X"].values, valid["Final_Y"].values)

        def metric(err: np.ndarray) -> Tuple[float, float]:
            rmse = float(np.sqrt(mean_squared_error(np.zeros_like(err), err)))
            mae = float(mean_absolute_error(np.zeros_like(err), err))
            return rmse, mae

        rmse_a, mae_a = metric(err_a)
        rmse_b, mae_b = metric(err_b)
        return {
            "A": {"RMSE_m": rmse_a, "MAE_m": mae_a, "Errors": err_a},
            "B": {"RMSE_m": rmse_b, "MAE_m": mae_b, "Errors": err_b},
        }

    @staticmethod
    def save_node_error_analysis(df: pd.DataFrame, out_path: str) -> pd.DataFrame:
        valid = df.dropna(subset=["Node_x", "Node_y", "True_X", "True_Y", "StepA_X", "StepA_Y", "Final_X", "Final_Y"]).copy()
        valid["StepA_Error_m"] = np.sqrt((valid["StepA_X"] - valid["True_X"]) ** 2 + (valid["StepA_Y"] - valid["True_Y"]) ** 2)
        valid["StepB_Error_m"] = np.sqrt((valid["Final_X"] - valid["True_X"]) ** 2 + (valid["Final_Y"] - valid["True_Y"]) ** 2)

        grouped = (
            valid.groupby(["Node_x", "Node_y"], as_index=False)[["StepA_Error_m", "StepB_Error_m"]]
            .mean()
            .sort_values("StepB_Error_m", ascending=False)
        )
        grouped.to_csv(out_path, index=False)
        return grouped

    @staticmethod
    def plot_cdf_comparison(err_a: np.ndarray, err_b: np.ndarray, out_path: str) -> None:
        plt.figure(figsize=(8, 5))
        for errs, label, color in [
            (err_a, "Step A: Pure Math", "tab:blue"),
            (err_b, "Step B: Final Hybrid", "tab:red"),
        ]:
            se = np.sort(errs)
            cdf = np.arange(1, len(se) + 1) / len(se)
            plt.plot(se, cdf, linewidth=2, label=label, color=color)
        plt.xlabel("Position Error (m)")
        plt.ylabel("Cumulative Probability")
        plt.title("V3 Error CDF Comparison (Step A vs Step B)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    @staticmethod
    def plot_multistage_map(df: pd.DataFrame, out_path: str) -> None:
        valid = df.dropna(subset=["True_X", "True_Y", "StepA_X", "StepA_Y", "Final_X", "Final_Y"])
        tx, ty = valid["True_X"].values, valid["True_Y"].values
        ax, ay = valid["StepA_X"].values, valid["StepA_Y"].values
        bx, by = valid["Final_X"].values, valid["Final_Y"].values

        plt.figure(figsize=(10, 6))
        plt.scatter(tx, ty, c="blue", s=22, alpha=0.85, label="True")
        plt.scatter(ax, ay, c="gray", s=20, alpha=0.75, label="Step A")
        plt.scatter(bx, by, c="red", s=22, alpha=0.85, label="Step B")

        # Correction vectors: Step A -> Step B
        plt.quiver(
            ax,
            ay,
            bx - ax,
            by - ay,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="darkred",
            alpha=0.45,
            width=0.002,
        )
        plt.xlim(0, 12)
        plt.ylim(0, 18)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("V3 Multi-stage Map (True / Step A / Step B)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.gca().set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()


def main() -> int:
    cfg = Config()
    localizer = WifiUwbFusionLocalizerV3(cfg)

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
    print(f"V3: 학습 {len(train)}행 ({train_med.name}) · 검증 {len(val)}행 ({val_med.name})")

    # Phase 1
    localizer.fit_uwb_imputer(train)
    train_p = localizer.add_imputed_uwb_features(train)
    val_p = localizer.add_imputed_uwb_features(val)

    # Phase 2: Step A (Pure Math)
    train_a = localizer.run_step_a_pure_math(train_p)
    val_a = localizer.run_step_a_pure_math(val_p)

    val_wifi_only = localizer.run_wifi_only_trilateration(val_p)
    err_wifi = WifiUwbFusionLocalizerV3.point_error(
        val_wifi_only["True_X"].values,
        val_wifi_only["True_Y"].values,
        val_wifi_only["WifiOnly_X"].values,
        val_wifi_only["WifiOnly_Y"].values,
    )
    fin_w = np.isfinite(err_wifi)
    ewf = err_wifi[fin_w]
    rmse_wifi = (
        float(np.sqrt(mean_squared_error(np.zeros_like(ewf), ewf))) if ewf.size else float("nan")
    )
    mae_wifi = float(mean_absolute_error(np.zeros_like(ewf), ewf)) if ewf.size else float("nan")
    print("=== Validation (Wi-Fi only 삼변, UWB 미사용 · V3 알고리즘) ===")
    print(f"Wi-Fi-only RMSE: {rmse_wifi:.4f} m | MAE: {mae_wifi:.4f} m")

    # Phase 3: Step B (Final Hybrid with 2D feature only)
    localizer.fit_phase3_knn(train_a)
    val_b = localizer.apply_step_b_final_hybrid(val_a)

    # Evaluation
    metrics = localizer.evaluate_step_a_b(val_b)
    print("=== V3 Step-by-Step Validation Performance ===")
    print(f"Step A [Pure Math]    RMSE: {metrics['A']['RMSE_m']:.4f} m | MAE: {metrics['A']['MAE_m']:.4f} m")
    print(f"Step B [Final Hybrid] RMSE: {metrics['B']['RMSE_m']:.4f} m | MAE: {metrics['B']['MAE_m']:.4f} m")

    pred_cols = ["Node_x", "Node_y", "True_X", "True_Y", "StepA_X", "StepA_Y", "Final_X", "Final_Y"]
    pred_path = os.path.join(output_dir, "v3_validation_predictions.csv")
    val_b[pred_cols].to_csv(pred_path, index=False)

    report_path = os.path.join(output_dir, "v3_step_rmse_report.csv")
    report_df = pd.DataFrame(
        [
            {"Step": "A", "Method": "Pure Math", "RMSE_m": metrics["A"]["RMSE_m"], "MAE_m": metrics["A"]["MAE_m"]},
            {"Step": "B", "Method": "Final Hybrid", "RMSE_m": metrics["B"]["RMSE_m"], "MAE_m": metrics["B"]["MAE_m"]},
        ]
    )
    report_df.to_csv(report_path, index=False)

    node_path = os.path.join(output_dir, "v3_node_error_analysis.csv")
    node_df = localizer.save_node_error_analysis(val_b, node_path)
    worst10 = node_df.head(10)
    print("\n=== Worst 10 Nodes by Step B Error ===")
    for _, r in worst10.iterrows():
        print(
            f"Node({int(r['Node_x'])}, {int(r['Node_y'])}) | "
            f"StepA: {r['StepA_Error_m']:.3f} m -> StepB: {r['StepB_Error_m']:.3f} m"
        )

    cdf_path = os.path.join(output_dir, "v3_validation_error_cdf_stepA_stepB.png")
    map_path = os.path.join(output_dir, "v3_validation_multistage_map.png")
    localizer.plot_cdf_comparison(metrics["A"]["Errors"], metrics["B"]["Errors"], cdf_path)
    localizer.plot_multistage_map(val_b, map_path)

    return 0


if __name__ == "__main__":
    from pathlib import Path

    from script_run_io import cli_entrypoint

    cli_entrypoint(Path(__file__), main, output_artifact_include_prefixes=("v3_",))
