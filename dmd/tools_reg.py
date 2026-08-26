"""Discovery, probing, and invocation of repo-provided tool scripts."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Optional

import yaml

from .types import ToolSpec

_TOOL_EXTS: tuple[str, ...] = (".py", ".sh")


def _executable_candidate(path: str) -> Optional[list[str]]:
    if not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext not in _TOOL_EXTS:
        return None
    if not os.access(path, os.X_OK):
        return None
    return [path]


def discover_tools(root: str) -> list[list[str]]:
    """List raw command vectors from a project's tools dir and tools.yaml."""
    candidates: list[list[str]] = []
    tools_dir = os.path.join(root, "tools")
    if os.path.isdir(tools_dir):
        for name in sorted(os.listdir(tools_dir)):
            full = os.path.join(tools_dir, name)
            cand = _executable_candidate(full)
            if cand is not None:
                candidates.append(cand)
    yaml_path = os.path.join(root, "tools.yaml")
    if os.path.isfile(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except (OSError, yaml.YAMLError):
            data = None
        if isinstance(data, dict):
            for _name, spec in data.items():
                if not isinstance(spec, dict):
                    continue
                cmd = spec.get("command")
                if (
                    isinstance(cmd, list)
                    and cmd
                    and all(isinstance(x, str) for x in cmd)
                ):
                    candidates.append(list(cmd))
    return candidates


def _derive_name(command: list[str]) -> str:
    if not command:
        return "tool"
    first = command[0]
    if "/" in first or os.sep in first:
        stem = os.path.splitext(os.path.basename(first))[0]
        return stem or "tool"
    return first


class ToolRegistry:
    """Probe and call repo-provided tool commands."""

    def __init__(self) -> None:
        self.tools: dict[str, ToolSpec] = {}
        self.rejected: list[tuple[list[str], str]] = []
        self._cache: dict[str, tuple[float, dict]] = {}
        self._cache_lock = asyncio.Lock()

    async def register(
        self,
        candidates: list[list[str]],
        *,
        probe_timeout_s: float = 10.0,
    ) -> list[ToolSpec]:
        """Probe each candidate with --json-probe; record successes as ToolSpec."""
        self.rejected = []
        for cmd in candidates:
            cmd_list = list(cmd)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd_list,
                    "--json-probe",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError as e:
                self.rejected.append((cmd_list, f"missing:{e}"))
                continue
            except Exception as e:
                self.rejected.append((cmd_list, f"spawn_error:{type(e).__name__}"))
                continue

            try:
                stdout_b, _stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=probe_timeout_s
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                self.rejected.append((cmd_list, "probe_timeout"))
                continue

            if proc.returncode != 0:
                self.rejected.append(
                    (cmd_list, f"returncode_{proc.returncode}")
                )
                continue

            try:
                text = stdout_b.decode("utf-8", errors="replace")
                obj = json.loads(text)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self.rejected.append(
                    (cmd_list, f"parse_error:{type(e).__name__}")
                )
                continue

            if not isinstance(obj, dict):
                self.rejected.append((cmd_list, "not_object"))
                continue

            try:
                timeout_s = float(obj.get("timeout_s", 15))
            except (TypeError, ValueError):
                timeout_s = 15.0
            try:
                cache_ttl_s = float(obj.get("cache_ttl_s", 300))
            except (TypeError, ValueError):
                cache_ttl_s = 300.0

            name = _derive_name(cmd_list)
            base_name = name
            n = 2
            while name in self.tools:
                name = f"{base_name}_{n}"
                n += 1

            spec = ToolSpec(
                name=name,
                command=cmd_list,
                description=str(obj.get("description", "")),
                output_schema_hint=json.dumps(obj)[:2000],
                timeout_s=timeout_s,
                cache_ttl_s=cache_ttl_s,
            )
            self.tools[name] = spec
        return list(self.tools.values())

    async def call(
        self, name: str, args: dict | None = None
    ) -> dict | None:
        """Invoke a registered tool with --call <json>; cache result by name."""
        spec = self.tools.get(name)
        if spec is None:
            return None
        now = time.monotonic()
        async with self._cache_lock:
            cached = self._cache.get(name)
            if cached is not None and (now - cached[0]) < spec.cache_ttl_s:
                return cached[1]

        try:
            proc = await asyncio.create_subprocess_exec(
                *spec.command,
                "--call",
                json.dumps(args or {}),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, PermissionError, OSError):
            return None
        except Exception:
            return None

        try:
            stdout_b, _stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=spec.timeout_s
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return None

        if proc.returncode != 0:
            return None
        try:
            text = stdout_b.decode("utf-8", errors="replace")
            obj = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(obj, dict):
            return None

        async with self._cache_lock:
            self._cache[name] = (time.monotonic(), obj)
        return obj
