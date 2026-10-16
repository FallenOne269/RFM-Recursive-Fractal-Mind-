"""Pytest run management for the RFM MCP server."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

OUTPUT_TAIL_CHARS = 12_000

PYTEST_EXIT_STATUS = {
    0: "passed",
    1: "failed",
    2: "interrupted",
    3: "internal_error",
    4: "usage_error",
    5: "no_tests_collected",
}


class RunError(Exception):
    """Raised when a test run cannot be started or found."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RunRecord:
    run_id: str
    command: list[str]
    target: Optional[str]
    keyword: Optional[str]
    timeout_seconds: int
    status: str = "running"
    started_at: str = field(default_factory=_now)
    finished_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    exit_code: Optional[int] = None
    summary: dict[str, int] = field(default_factory=dict)
    failures: list[dict[str, str]] = field(default_factory=list)
    output: str = ""

    def to_dict(self, include_output: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "run_id": self.run_id,
            "status": self.status,
            "command": " ".join(self.command),
            "target": self.target,
            "keyword": self.keyword,
            "timeout_seconds": self.timeout_seconds,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "exit_code": self.exit_code,
            "summary": self.summary,
            "failures": self.failures,
        }
        if include_output:
            data["output"] = self.output[-OUTPUT_TAIL_CHARS:]
        return data


def parse_junit(path: Path) -> tuple[dict[str, int], list[dict[str, str]]]:
    """Return (summary counts, failure details) from a pytest JUnit XML report."""
    if not path.is_file():
        return {}, []
    root = ET.parse(path).getroot()
    counts = {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    failures: list[dict[str, str]] = []
    for case in root.iter("testcase"):
        counts["total"] += 1
        test_id = f"{case.get('classname', '')}::{case.get('name', '')}"
        outcome = "passed"
        for child in case:
            if child.tag in ("failure", "error"):
                outcome = "failed" if child.tag == "failure" else "errors"
                failures.append(
                    {
                        "test": test_id,
                        "kind": child.tag,
                        "message": (child.get("message") or "").strip()[:2000],
                        "details": (child.text or "").strip()[-4000:],
                    }
                )
                break
            if child.tag == "skipped":
                outcome = "skipped"
        counts[outcome] += 1
    return counts, failures


class PytestRunner:
    """Launches pytest in a subprocess and keeps an in-memory run history."""

    def __init__(
        self,
        repo_root: Path,
        python: str = sys.executable,
        default_timeout: int = 600,
        max_history: int = 50,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.python = python
        self.default_timeout = default_timeout
        self.max_history = max_history
        self._runs: OrderedDict[str, RunRecord] = OrderedDict()
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def validate_target(self, target: Optional[str]) -> Optional[str]:
        if not target:
            return None
        if target.startswith("-"):
            raise RunError("target must be a test path or node id, not a pytest option")
        file_part = target.split("::", 1)[0]
        resolved = (self.repo_root / file_part).resolve()
        if not resolved.is_relative_to(self.repo_root):
            raise RunError(f"target {target!r} is outside the repository")
        if not resolved.exists():
            raise RunError(f"target {file_part!r} does not exist")
        return target

    def _base_command(self) -> list[str]:
        return [self.python, "-m", "pytest", "-p", "no:cacheprovider"]

    async def collect(self, target: Optional[str] = None) -> list[str]:
        """Return the node ids pytest would run for *target*."""
        target = self.validate_target(target)
        cmd = self._base_command() + ["--collect-only", "-q"]
        if target:
            cmd.append(target)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.repo_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(
                proc.communicate(), timeout=self.default_timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RunError("test collection timed out") from None
        text = out.decode(errors="replace")
        if proc.returncode not in (0, 5):
            raise RunError(
                f"test collection failed (exit {proc.returncode}):\n{text[-4000:]}"
            )
        return [line for line in text.splitlines() if "::" in line and " " not in line]

    def start(
        self,
        target: Optional[str] = None,
        keyword: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> RunRecord:
        """Start a pytest run in the background and return its record."""
        running = [r.run_id for r in self._runs.values() if r.status == "running"]
        if running:
            raise RunError(f"test run {running[0]} is still in progress")
        target = self.validate_target(target)
        timeout = timeout_seconds or self.default_timeout
        if timeout <= 0:
            raise RunError("timeout_seconds must be positive")

        workdir = Path(tempfile.mkdtemp(prefix="rfai-mcp-run-"))
        junit = workdir / "junit.xml"
        cmd = self._base_command() + ["-rA", f"--junitxml={junit}"]
        if keyword:
            cmd += ["-k", keyword]
        if target:
            cmd.append(target)

        record = RunRecord(
            run_id=uuid.uuid4().hex[:12],
            command=cmd,
            target=target,
            keyword=keyword,
            timeout_seconds=timeout,
        )
        self._remember(record)
        self._tasks[record.run_id] = asyncio.create_task(
            self._execute(record, workdir, junit)
        )
        return record

    async def run(
        self,
        target: Optional[str] = None,
        keyword: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> RunRecord:
        """Run pytest and wait for it to finish."""
        record = self.start(target, keyword, timeout_seconds)
        await self.wait(record.run_id)
        return record

    async def wait(self, run_id: str) -> RunRecord:
        record = self.get(run_id)
        task = self._tasks.get(run_id)
        if task is not None:
            await asyncio.shield(task)
        return record

    def get(self, run_id: str) -> RunRecord:
        try:
            return self._runs[run_id]
        except KeyError:
            raise RunError(f"unknown run_id {run_id!r}") from None

    def list(self, limit: int = 20) -> list[RunRecord]:
        return list(reversed(self._runs.values()))[:limit]

    def _remember(self, record: RunRecord) -> None:
        self._runs[record.run_id] = record
        while len(self._runs) > self.max_history:
            old_id, old = next(iter(self._runs.items()))
            if old.status == "running":
                break
            self._runs.pop(old_id)

    async def _execute(self, record: RunRecord, workdir: Path, junit: Path) -> None:
        log_path = workdir / "output.log"
        start = time.monotonic()
        try:
            with log_path.open("wb") as log:
                proc = await asyncio.create_subprocess_exec(
                    *record.command,
                    cwd=self.repo_root,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                try:
                    record.exit_code = await asyncio.wait_for(
                        proc.wait(), timeout=record.timeout_seconds
                    )
                    record.status = PYTEST_EXIT_STATUS.get(record.exit_code, "error")
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    record.status = "timeout"
            record.output = log_path.read_text(errors="replace")[-OUTPUT_TAIL_CHARS:]
            record.summary, record.failures = parse_junit(junit)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - surface any launcher failure to the client
            record.status = "error"
            record.output += f"\n[rfai-mcp] failed to run pytest: {exc!r}"
        finally:
            record.duration_seconds = round(time.monotonic() - start, 3)
            record.finished_at = _now()
            self._tasks.pop(record.run_id, None)
            shutil.rmtree(workdir, ignore_errors=True)
