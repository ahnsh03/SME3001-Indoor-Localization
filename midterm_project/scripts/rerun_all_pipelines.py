"""
Re-run all indoor fusion pipeline versions once; write batch_run_manifest.json.
Run from midterm_project root: py -3 scripts/rerun_all_pipelines.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "outputs"
LOG = OUT / "batch_rerun.log"

# Faster Optuna for batch reproducibility (override in shell if needed)
os.environ.setdefault("V15_OPTUNA_TRIALS", "12")
os.environ.setdefault("V13_OPTUNA_TRIALS", "24")
os.environ.setdefault("V12_TURBO_OPTUNA_TRIALS", "24")
os.environ.setdefault("V15_TUNING_PROFILE", "defensive")

PIPELINES: list[tuple[str, list[str]]] = [
    ("v1", ["indoor_fusion_pipeline_v1.py"]),
    ("v2", ["indoor_fusion_pipeline_v2.py"]),
    ("v3", ["indoor_fusion_pipeline_v3.py"]),
    ("v4", ["indoor_fusion_pipeline_v4.py"]),
    ("v5", ["indoor_fusion_pipeline_v5.py"]),
    ("v6", ["indoor_fusion_pipeline_v6.py"]),
    ("v7", ["indoor_fusion_pipeline_v7.py"]),
    ("v8", ["indoor_fusion_pipeline_v8.py"]),
    ("v9", ["indoor_fusion_pipeline_v9.py"]),
    ("v9_strict", ["indoor_fusion_pipeline_v9_strict.py"]),
    ("v10", ["indoor_fusion_pipeline_v10.py"]),
    ("v10_opt", ["indoor_fusion_pipeline_v10_optimized.py"]),
    ("v11", ["indoor_fusion_pipeline_v11.py"]),
    ("v12", ["indoor_fusion_pipeline_v12.py"]),
    ("v12_strict", ["indoor_fusion_pipeline_v12_strict.py"]),
    ("v12_strict_parallel", ["indoor_fusion_pipeline_v12_strict_parallel.py"]),
    ("v12_fast", ["indoor_fusion_pipeline_v12_fast.py"]),
    ("v12_fast2", ["indoor_fusion_pipeline_v12_fast2.py"]),
    ("v12_turbo", ["indoor_fusion_pipeline_v12_turbo.py", "--no-plots"]),
    ("v13", ["indoor_fusion_pipeline_v13.py", "--no-plots"]),
    ("v13_fix", ["indoor_fusion_pipeline_v13_fix.py", "--no-plots"]),
    ("v14", ["indoor_fusion_pipeline_v14.py", "--no-plots"]),
    ("v15", ["indoor_fusion_pipeline_v15.py", "--no-plots"]),
    ("v16", ["indoor_fusion_pipeline_v16.py", "--no-plots"]),
    ("JWT", ["indoor_fusion_pipeline_JWT.py"]),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    py = sys.executable
    with LOG.open("w", encoding="utf-8") as logf:
        logf.write(f"cwd={ROOT}\npython={py}\n\n")
        for name, args in PIPELINES:
            script = SCRIPTS / args[0]
            extra = args[1:]
            if not script.is_file():
                rec = {"name": name, "status": "missing", "sec": 0.0}
                results.append(rec)
                logf.write(f"[SKIP] {name} missing {script}\n")
                continue
            cmd = [py, str(script), *extra]
            logf.write(f"\n{'='*60}\n[{name}] {' '.join(cmd)}\n")
            logf.flush()
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=3600,
                )
                sec = time.perf_counter() - t0
                logf.write(proc.stdout)
                if proc.stderr:
                    logf.write("\n--- stderr ---\n")
                    logf.write(proc.stderr)
                status = "ok" if proc.returncode == 0 else f"exit_{proc.returncode}"
            except subprocess.TimeoutExpired as e:
                sec = time.perf_counter() - t0
                logf.write(str(e.stdout or "") + str(e.stderr or ""))
                status = "timeout"
            except Exception as e:
                sec = time.perf_counter() - t0
                logf.write(f"ERROR: {e}\n")
                status = "error"
            rec = {"name": name, "status": status, "sec": round(sec, 2)}
            results.append(rec)
            logf.write(f"\n[{name}] done status={status} sec={sec:.1f}\n")
            logf.flush()
            print(f"[{name}] {status} ({sec:.1f}s)", flush=True)

    manifest = {"pipelines": results, "root": str(ROOT)}
    (OUT / "batch_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    failed = [r for r in results if r["status"] != "ok"]
    print(f"Finished {len(results)} runs, failed={len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
