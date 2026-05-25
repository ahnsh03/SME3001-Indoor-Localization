"""
스크립트 실행 시 콘솔 로그를 outputs/run_reports/<run_id>/run.log 로 복제하고,
manifest.json 에 argv·환경·산출물 목록·바이트 수를 기록한다.

데이터 처리 로직은 건드리지 않으며, outputs/ 직하위 파일만 산출물로 스캔한다
(run_reports 하위는 제외).
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

JsonDict = Dict[str, Any]


def project_root_from_script(script_path: Path) -> Path:
    """scripts/foo.py → 프로젝트 루트."""
    return Path(script_path).resolve().parent.parent


class _TeeTextStream:
    """stdout/stderr 를 원 스트림과 로그 파일에 동시 기록."""

    def __init__(self, primary: Any, log_fp: Any) -> None:
        self._primary = primary
        self._log = log_fp

    @property
    def encoding(self) -> str:
        enc = getattr(self._primary, "encoding", None) or "utf-8"
        return str(enc)

    def write(self, data: str) -> int:
        if isinstance(data, bytes):
            data = data.decode(self.encoding, errors="replace")
        self._log.write(data)
        self._log.flush()
        try:
            n = self._primary.write(data)
            return int(n) if n is not None else len(data)
        except UnicodeEncodeError:
            enc = getattr(self._primary, "encoding", None) or "utf-8"
            safe = data.encode(enc, errors="replace").decode(enc, errors="replace")
            n = self._primary.write(safe)
            return int(n) if n is not None else len(safe)

    def flush(self) -> None:
        self._log.flush()
        self._primary.flush()

    def isatty(self) -> bool:
        return bool(self._primary.isatty())


class RunSession:
    """한 번의 스크립트 실행에 대한 로그·manifest 세션."""

    def __init__(self, project_root: Path, script_stem: str) -> None:
        self.project_root = Path(project_root)
        self.script_stem = script_stem
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{script_stem}_{ts}"
        self.outputs_root = self.project_root / "outputs"
        self.run_dir = self.outputs_root / "run_reports" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "run.log"
        self.manifest_path = self.run_dir / "manifest.json"
        self._artifacts: List[JsonDict] = []
        self._t0 = time.perf_counter()
        self._utc_iso = datetime.now(timezone.utc).isoformat()
        self._orig_out = sys.stdout
        self._orig_err = sys.stderr
        self._log_fp = self.log_path.open("w", encoding="utf-8", newline="\n")
        self._log_fp.write(
            f"# run_id={self.run_id}\n# started_utc={self._utc_iso}\n# cwd={os.getcwd()}\n"
            f"# argv={sys.argv!r}\n\n"
        )
        self._log_fp.flush()
        sys.stdout = _TeeTextStream(self._orig_out, self._log_fp)  # type: ignore[assignment]
        sys.stderr = _TeeTextStream(self._orig_err, self._log_fp)  # type: ignore[assignment]

    def register_path(self, path: Path, *, note: str = "") -> None:
        """수동으로 산출물 한 경로 등록(프로젝트 루트 기준 상대 경로·bytes)."""
        p = Path(path).resolve()
        try:
            rel = str(p.relative_to(self.project_root))
        except ValueError:
            rel = str(p)
        row: JsonDict = {"path": rel, "kind": "file", "note": note}
        if p.is_file():
            row["bytes"] = p.stat().st_size
        self._artifacts.append(row)

    def _scan_output_artifacts(
        self,
        include_prefixes: Sequence[str],
        exclude_prefixes: Sequence[str],
    ) -> None:
        if not include_prefixes:
            return
        self.outputs_root.mkdir(parents=True, exist_ok=True)
        for p in sorted(self.outputs_root.iterdir()):
            if not p.is_file():
                continue
            name = p.name
            if not any(name.startswith(pref) for pref in include_prefixes):
                continue
            if exclude_prefixes and any(name.startswith(pref) for pref in exclude_prefixes):
                continue
            self.register_path(p)

    def _print_artifact_table(self) -> None:
        if not self._artifacts:
            print("\n[산출물] outputs/ 직하위에서 이번 규칙으로 매칭된 파일 없음.", flush=True)
            return
        print("\n" + "=" * 72, flush=True)
        print(f"[{self.script_stem}] 산출물 요약 (총 {len(self._artifacts)}개)", flush=True)
        print("-" * 72, flush=True)
        for row in sorted(self._artifacts, key=lambda r: r.get("path", "")):
            rel = row.get("path", "")
            sz = row.get("bytes")
            extra = f"  ({sz:,} bytes)" if isinstance(sz, int) else ""
            print(f"  {rel}{extra}", flush=True)
        print("=" * 72, flush=True)

    def finish(
        self,
        exit_code: int,
        *,
        output_artifact_include_prefixes: Sequence[str] = (),
        output_artifact_exclude_prefixes: Sequence[str] = (),
        extra_manifest: Optional[JsonDict] = None,
    ) -> None:
        self._scan_output_artifacts(output_artifact_include_prefixes, output_artifact_exclude_prefixes)
        elapsed = time.perf_counter() - self._t0
        manifest: JsonDict = {
            "schema": "midterm_script_run_v1",
            "run_id": self.run_id,
            "script_stem": self.script_stem,
            "started_utc": self._utc_iso,
            "cwd": os.getcwd(),
            "argv": sys.argv,
            "python_version": sys.version,
            "platform": sys.platform,
            "exit_code": int(exit_code),
            "elapsed_sec": round(float(elapsed), 4),
            "outputs_dir": str(self.outputs_root),
            "run_log": str(self.log_path.relative_to(self.project_root)),
            "artifacts": sorted(self._artifacts, key=lambda r: r.get("path", "")),
        }
        if extra_manifest:
            manifest["extra"] = extra_manifest

        self._print_artifact_table()

        sys.stdout = self._orig_out
        sys.stderr = self._orig_err
        self._log_fp.flush()
        self._log_fp.close()

        self.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[run_reports] manifest 저장: {self.manifest_path}", flush=True)
        print(f"[run_reports] 전체 로그: {self.log_path}", flush=True)


def cli_entrypoint(
    script_path: Path,
    main_fn: Callable[[], Optional[Union[int, bool]]],
    *,
    output_artifact_include_prefixes: Sequence[str] = (),
    output_artifact_exclude_prefixes: Sequence[str] = (),
    extra_manifest: Optional[JsonDict] = None,
) -> None:
    """표준 __main__ 블록: RunSession + main 호출 + manifest."""
    root = project_root_from_script(script_path)
    stem = Path(script_path).stem
    session = RunSession(root, stem)
    code = 1
    try:
        result = main_fn()
        if result is None:
            code = 0
        elif isinstance(result, bool):
            code = 0 if result else 1
        else:
            code = int(result)
    except SystemExit as e:
        c = e.code
        if c is None:
            code = 0
        elif isinstance(c, int):
            code = c
        else:
            code = 1
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        session.finish(
            code,
            output_artifact_include_prefixes=output_artifact_include_prefixes,
            output_artifact_exclude_prefixes=output_artifact_exclude_prefixes,
            extra_manifest=extra_manifest,
        )
    raise SystemExit(code)
