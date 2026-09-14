"""
Reusable pipeline summary/error reporter.

Standard-library only. Designed to be copied into any Python repo and used
across unrelated pipeline scripts while keeping one consistent report format.

Typical use:

    from pipeline_reporter import PipelineReporter

    with PipelineReporter(
        script=__file__,
        pipeline="cfb",
        stage="00_intake",
        output_dir="reports",
        season=2026,
        week=2,
    ) as report:
        report.add_input("input.csv")
        report.set_detail("games_fetched", 85)

        # Run normal script logic here.

        report.add_output("output.csv")
        report.set_rows(rows_in=85, rows_out=82)

If the script finishes normally, the report is written with status SUCCESS.
If an unhandled exception occurs, the exception and traceback are recorded,
the report is written with status FAILED, and the exception is re-raised.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "1.0"
VALID_STATUSES = {"SUCCESS", "WARNING", "FAILED"}


class PipelineReporter:
    def __init__(
        self,
        *,
        script: str,
        pipeline: str,
        output_dir: str | os.PathLike[str],
        stage: str | None = None,
        league: str | None = None,
        season: int | str | None = None,
        week: int | str | None = None,
        run_id: str | None = None,
        extra_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.script = Path(script).name
        self.script_path = str(Path(script))
        self.pipeline = pipeline
        self.stage = stage
        self.league = league
        self.season = season
        self.week = week
        self.run_id = run_id or os.getenv("GITHUB_RUN_ID") or uuid.uuid4().hex
        self.output_dir = Path(output_dir)

        self._started_monotonic = time.monotonic()
        self._started_at = datetime.now(timezone.utc)

        self._status = "SUCCESS"
        self._warnings: list[dict[str, Any]] = []
        self._errors: list[dict[str, Any]] = []
        self._inputs: list[str] = []
        self._outputs: list[str] = []
        self._details: dict[str, Any] = {}
        self._rows_in: int | None = None
        self._rows_out: int | None = None
        self._extra_context = dict(extra_context or {})
        self._written = False
        self.report_path: Path | None = None

    def __enter__(self) -> "PipelineReporter":
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> bool:
        if exc_value is not None:
            self.record_exception(exc_value, exc_tb)
            self.write_report(status="FAILED")
            return False

        if self._errors:
            status = "FAILED"
        elif self._warnings:
            status = "WARNING"
        else:
            status = "SUCCESS"

        self.write_report(status=status)
        return False

    def add_input(self, path: str | os.PathLike[str]) -> None:
        value = str(path)
        if value not in self._inputs:
            self._inputs.append(value)

    def add_output(self, path: str | os.PathLike[str]) -> None:
        value = str(path)
        if value not in self._outputs:
            self._outputs.append(value)

    def set_rows(
        self,
        *,
        rows_in: int | None = None,
        rows_out: int | None = None,
    ) -> None:
        if rows_in is not None:
            self._rows_in = int(rows_in)
        if rows_out is not None:
            self._rows_out = int(rows_out)

    def set_detail(self, key: str, value: Any) -> None:
        self._details[str(key)] = value

    def update_details(self, values: Mapping[str, Any]) -> None:
        self._details.update(dict(values))

    def warning(self, message: str, **details: Any) -> None:
        self._warnings.append(
            {
                "timestamp_utc": _utc_now_iso(),
                "message": str(message),
                "details": details or None,
            }
        )
        if self._status == "SUCCESS":
            self._status = "WARNING"

    def error(
        self,
        message: str,
        *,
        error_type: str | None = None,
        traceback_text: str | None = None,
        **details: Any,
    ) -> None:
        self._errors.append(
            {
                "timestamp_utc": _utc_now_iso(),
                "type": error_type,
                "message": str(message),
                "traceback": traceback_text,
                "details": details or None,
            }
        )
        self._status = "FAILED"

    def record_exception(self, exc: BaseException, tb=None) -> None:
        if tb is None:
            tb = exc.__traceback__

        self.error(
            str(exc),
            error_type=type(exc).__name__,
            traceback_text="".join(
                traceback.format_exception(type(exc), exc, tb)
            ),
        )

    def write_report(
        self,
        *,
        status: str | None = None,
        exit_code: int | None = None,
    ) -> Path:
        final_status = (status or self._status).upper()

        if final_status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status {final_status!r}. "
                f"Expected one of: {sorted(VALID_STATUSES)}"
            )

        if self._errors:
            final_status = "FAILED"
        elif self._warnings and final_status == "SUCCESS":
            final_status = "WARNING"

        finished_at = datetime.now(timezone.utc)
        duration_seconds = round(time.monotonic() - self._started_monotonic, 3)

        report = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_utc": finished_at.isoformat(),
            "run_id": self.run_id,
            "pipeline": self.pipeline,
            "league": self.league,
            "stage": self.stage,
            "script": self.script,
            "script_path": self.script_path,
            "status": final_status,
            "exit_code": exit_code,
            "started_at_utc": self._started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": duration_seconds,
            "environment": {
                "hostname": socket.gethostname(),
                "python_version": sys.version.split()[0],
                "github_run_id": os.getenv("GITHUB_RUN_ID"),
                "github_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
                "github_workflow": os.getenv("GITHUB_WORKFLOW"),
                "github_job": os.getenv("GITHUB_JOB"),
                "github_ref_name": os.getenv("GITHUB_REF_NAME"),
                "github_sha": os.getenv("GITHUB_SHA"),
            },
            "context": {
                "season": self.season,
                "week": self.week,
                **self._extra_context,
            },
            "inputs": self._inputs,
            "outputs": self._outputs,
            "rows_in": self._rows_in,
            "rows_out": self._rows_out,
            "warning_count": len(self._warnings),
            "error_count": len(self._errors),
            "warnings": self._warnings,
            "errors": self._errors,
            "details": self._details,
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)

        safe_pipeline = _safe_filename(self.pipeline)
        safe_stage = _safe_filename(self.stage or "general")
        safe_script = _safe_filename(Path(self.script).stem)
        filename = f"{safe_pipeline}__{safe_stage}__{safe_script}.json"

        destination = self.output_dir / filename
        temp_path = destination.with_suffix(destination.suffix + ".tmp")

        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, sort_keys=False, default=str)
            handle.write("\n")

        os.replace(temp_path, destination)

        self._status = final_status
        self._written = True
        self.report_path = destination
        return destination


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(value: str) -> str:
    cleaned = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "report"


__all__ = ["PipelineReporter", "SCHEMA_VERSION"]
