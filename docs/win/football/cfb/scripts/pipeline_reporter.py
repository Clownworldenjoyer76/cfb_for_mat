#!/usr/bin/env python3
"""
Reusable pipeline summary/error reporter.

The caller supplies only:
- script
- stage
- report_root
- optional pipeline/league/season/week/context

Reports are written automatically as:

    {report_root}/{stage}/{script_stem}.json

Example:

    PipelineReporter(
        script=__file__,
        stage="00_intake",
        report_root="docs/project/errors",
    )

writes:

    docs/project/errors/00_intake/build_travel.json

Each execution replaces the prior report for that script so the file always
describes the most recent execution.

Unhandled exceptions are recorded with their traceback, the report is written
with FAILED status, and the original exception is re-raised.
"""

from __future__ import annotations

import json
import math
import os
import socket
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union


SCHEMA_VERSION = "2.0"
VALID_STATUSES = {"SUCCESS", "WARNING", "FAILED"}


class PipelineReporter:
    def __init__(
        self,
        *,
        script: Union[str, os.PathLike[str]],
        stage: str,
        report_root: Union[str, os.PathLike[str]],
        pipeline: Optional[str] = None,
        league: Optional[str] = None,
        season: Optional[Union[int, str]] = None,
        week: Optional[Union[int, str]] = None,
        run_id: Optional[str] = None,
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        stage_text = str(stage).strip()
        if not stage_text:
            raise ValueError("stage must be a non-empty string")

        script_path = Path(script)

        self.script = script_path.name
        self.script_stem = script_path.stem
        self.script_path = str(script_path)

        self.stage = stage_text
        self.pipeline = _clean_optional(pipeline)
        self.league = _clean_optional(league)
        self.season = season
        self.week = week

        self.report_root = Path(report_root)

        self.github_run_id = _clean_optional(
            os.getenv("GITHUB_RUN_ID")
        )
        self.github_run_attempt = _clean_optional(
            os.getenv("GITHUB_RUN_ATTEMPT")
        )

        self.run_id = (
            str(run_id).strip()
            if run_id is not None and str(run_id).strip()
            else _build_run_id(
                github_run_id=self.github_run_id,
                github_run_attempt=self.github_run_attempt,
            )
        )

        safe_stage = _safe_filename(self.stage)
        safe_script = _safe_filename(self.script_stem)

        self.output_dir = self.report_root / safe_stage
        self.report_path = self.output_dir / f"{safe_script}.json"

        self._started_monotonic = time.monotonic()
        self._started_at = datetime.now(timezone.utc)

        self._status = "SUCCESS"

        self._warnings: list[dict[str, Any]] = []
        self._errors: list[dict[str, Any]] = []

        self._inputs: list[str] = []
        self._outputs: list[str] = []

        self._details: dict[str, Any] = {}
        self._extra_context = dict(extra_context or {})

        self._rows_in: Optional[int] = None
        self._rows_out: Optional[int] = None

        self._written = False

    def __enter__(self) -> "PipelineReporter":
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> bool:
        if exc_value is not None:
            self.record_exception(exc_value, exc_tb)

            try:
                self.write_report(
                    status="FAILED",
                    exit_code=1,
                )
            except Exception as report_exc:
                _emit_reporting_failure(
                    report_path=self.report_path,
                    report_exception=report_exc,
                    original_exception=exc_value,
                )

            return False

        if self._errors:
            status = "FAILED"
            exit_code = 1
        elif self._warnings:
            status = "WARNING"
            exit_code = 0
        else:
            status = "SUCCESS"
            exit_code = 0

        self.write_report(
            status=status,
            exit_code=exit_code,
        )

        return False

    @property
    def status(self) -> str:
        return self._status

    @property
    def warning_count(self) -> int:
        return len(self._warnings)

    @property
    def error_count(self) -> int:
        return len(self._errors)

    @property
    def written(self) -> bool:
        return self._written

    def add_input(
        self,
        path: Union[str, os.PathLike[str]],
    ) -> None:
        value = str(path)

        if value not in self._inputs:
            self._inputs.append(value)

    def add_output(
        self,
        path: Union[str, os.PathLike[str]],
    ) -> None:
        value = str(path)

        if value not in self._outputs:
            self._outputs.append(value)

    def set_rows(
        self,
        *,
        rows_in: Optional[int] = None,
        rows_out: Optional[int] = None,
    ) -> None:
        if rows_in is not None:
            self._rows_in = _validate_nonnegative_int(
                rows_in,
                "rows_in",
            )

        if rows_out is not None:
            self._rows_out = _validate_nonnegative_int(
                rows_out,
                "rows_out",
            )

    def set_detail(
        self,
        key: str,
        value: Any,
    ) -> None:
        key_text = str(key).strip()

        if not key_text:
            raise ValueError("detail key must not be blank")

        self._details[key_text] = value

    def update_details(
        self,
        values: Mapping[str, Any],
    ) -> None:
        for key, value in values.items():
            self.set_detail(str(key), value)

    def warning(
        self,
        message: str,
        **details: Any,
    ) -> None:
        message_text = str(message).strip()

        if not message_text:
            raise ValueError("warning message must not be blank")

        self._warnings.append(
            {
                "timestamp_utc": _utc_now_iso(),
                "message": message_text,
                "details": details or None,
            }
        )

        if self._status == "SUCCESS":
            self._status = "WARNING"

    def error(
        self,
        message: str,
        *,
        error_type: Optional[str] = None,
        traceback_text: Optional[str] = None,
        **details: Any,
    ) -> None:
        message_text = (
            str(message).strip()
            or "Unspecified error"
        )

        self._errors.append(
            {
                "timestamp_utc": _utc_now_iso(),
                "type": _clean_optional(error_type),
                "message": message_text,
                "traceback": traceback_text,
                "details": details or None,
            }
        )

        self._status = "FAILED"

    def record_exception(
        self,
        exc: BaseException,
        tb=None,
    ) -> None:
        if tb is None:
            tb = exc.__traceback__

        traceback_text = "".join(
            traceback.format_exception(
                type(exc),
                exc,
                tb,
            )
        )

        self.error(
            str(exc) or type(exc).__name__,
            error_type=type(exc).__name__,
            traceback_text=traceback_text,
        )

    def write_report(
        self,
        *,
        status: Optional[str] = None,
        exit_code: Optional[int] = None,
    ) -> Path:
        final_status = (
            status or self._status
        ).upper().strip()

        if final_status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status {final_status!r}. "
                f"Expected one of: {sorted(VALID_STATUSES)}"
            )

        if self._errors:
            final_status = "FAILED"
        elif (
            self._warnings
            and final_status == "SUCCESS"
        ):
            final_status = "WARNING"

        if exit_code is None:
            exit_code = (
                1
                if final_status == "FAILED"
                else 0
            )
        else:
            exit_code = int(exit_code)

        if (
            final_status == "FAILED"
            and exit_code == 0
        ):
            exit_code = 1

        finished_at = datetime.now(timezone.utc)

        duration_seconds = round(
            max(
                0.0,
                time.monotonic()
                - self._started_monotonic,
            ),
            3,
        )

        report = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": final_status,
            "exit_code": exit_code,
            "pipeline": self.pipeline,
            "league": self.league,
            "stage": self.stage,
            "script": self.script,
            "script_path": self.script_path,
            "report_path": str(self.report_path),
            "started_at_utc": (
                self._started_at.isoformat()
            ),
            "finished_at_utc": (
                finished_at.isoformat()
            ),
            "duration_seconds": duration_seconds,
            "environment": {
                "hostname": socket.gethostname(),
                "python_version": (
                    sys.version.split()[0]
                ),
                "python_executable": sys.executable,
                "platform": sys.platform,
                "process_id": os.getpid(),
                "working_directory": os.getcwd(),
                "github_run_id": (
                    self.github_run_id
                ),
                "github_run_attempt": (
                    self.github_run_attempt
                ),
                "github_workflow": _clean_optional(
                    os.getenv("GITHUB_WORKFLOW")
                ),
                "github_job": _clean_optional(
                    os.getenv("GITHUB_JOB")
                ),
                "github_ref_name": _clean_optional(
                    os.getenv("GITHUB_REF_NAME")
                ),
                "github_sha": _clean_optional(
                    os.getenv("GITHUB_SHA")
                ),
                "github_actor": _clean_optional(
                    os.getenv("GITHUB_ACTOR")
                ),
                "github_repository": (
                    _clean_optional(
                        os.getenv(
                            "GITHUB_REPOSITORY"
                        )
                    )
                ),
            },
            "context": {
                "season": self.season,
                "week": self.week,
                **self._extra_context,
            },
            "inputs": list(self._inputs),
            "outputs": list(self._outputs),
            "rows_in": self._rows_in,
            "rows_out": self._rows_out,
            "warning_count": len(
                self._warnings
            ),
            "error_count": len(
                self._errors
            ),
            "warnings": list(
                self._warnings
            ),
            "errors": list(
                self._errors
            ),
            "details": dict(
                self._details
            ),
        }

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path = self.report_path.with_name(
            (
                f".{self.report_path.name}."
                f"{uuid.uuid4().hex}.tmp"
            )
        )

        try:
            with temp_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                json.dump(
                    report,
                    handle,
                    indent=2,
                    sort_keys=False,
                    ensure_ascii=False,
                    allow_nan=False,
                    default=_json_default,
                )

                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(
                temp_path,
                self.report_path,
            )

        except Exception:
            try:
                temp_path.unlink(
                    missing_ok=True
                )
            except OSError:
                pass

            raise

        self._status = final_status
        self._written = True

        return self.report_path


def _build_run_id(
    *,
    github_run_id: Optional[str],
    github_run_attempt: Optional[str],
) -> str:
    if github_run_id:
        if github_run_attempt:
            return (
                f"github-{github_run_id}"
                f"-attempt-{github_run_attempt}"
            )

        return f"github-{github_run_id}"

    return uuid.uuid4().hex


def _validate_nonnegative_int(
    value: Any,
    field: str,
) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"{field} must be a "
            "non-negative integer"
        )

    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} must be a "
            "non-negative integer"
        ) from exc

    if converted < 0:
        raise ValueError(
            f"{field} must be a "
            "non-negative integer"
        )

    return converted


def _json_default(
    value: Any,
) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, set):
        return sorted(
            value,
            key=str,
        )

    if hasattr(value, "item"):
        try:
            scalar = value.item()

            if (
                isinstance(scalar, float)
                and not math.isfinite(scalar)
            ):
                return str(scalar)

            return scalar

        except (TypeError, ValueError, OverflowError):
            pass

    return str(value)


def _clean_optional(
    value: Any,
) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    return text or None


def _utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _safe_filename(
    value: str,
) -> str:
    cleaned: list[str] = []

    for char in str(value):
        if (
            char.isalnum()
            or char in {"-", "_"}
        ):
            cleaned.append(char)
        else:
            cleaned.append("_")

    return (
        "".join(cleaned).strip("_")
        or "report"
    )


def _emit_reporting_failure(
    *,
    report_path: Path,
    report_exception: BaseException,
    original_exception: BaseException,
) -> None:
    try:
        sys.stderr.write(
            "\nPIPELINE REPORTING FAILURE\n"
            f"Report path: {report_path}\n"
            "Reporting error: "
            f"{type(report_exception).__name__}: "
            f"{report_exception}\n"
            "Original script error preserved: "
            f"{type(original_exception).__name__}: "
            f"{original_exception}\n"
        )

        sys.stderr.flush()

    except (AttributeError, OSError, UnicodeError, ValueError):
        pass


__all__ = [
    "PipelineReporter",
    "SCHEMA_VERSION",
]
