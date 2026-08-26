"""Tests for dmd.tools_reg — real subprocess scripts in tmp_path.

Each tool script is a Python file with a real `#!/usr/bin/env python3` shebang,
chmod'd 0o755, and dispatched on argv[1] (--json-probe | --call). Side-effects
(e.g. process spawn counter) are recorded by appending to a file.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

import pytest
import yaml

from dmd.tools_reg import ToolRegistry, discover_tools


def write_tool_script(
    tools_dir: Path,
    name: str,
    *,
    probe: Optional[dict] = None,
    side_effect: Optional[Path] = None,
    sleep_s: float = 0.0,
    exit_code: int = 0,
    bad_json: bool = False,
    executable: bool = True,
) -> Path:
    """Write a Python tool script with shebang; chmod to taste.

    - side_effect: file path the script appends 'spawn\\n' to on every invocation.
    - sleep_s: seconds the script sleeps BEFORE processing argv (for timeout test).
    - exit_code: nonzero exit on both --json-probe and --call (default 0).
    - bad_json: print garbage for both --json-probe and --call (still exit 0).
    """
    path = tools_dir / name
    lines: list[str] = [
        "#!/usr/bin/env python3",
        "import json, sys, time",
    ]
    if side_effect is not None:
        # Hardcode the side-effect path so the test can read it back deterministically.
        se_path = str(side_effect).replace("\\", "\\\\")
        lines.append(f'with open("{se_path}", "a") as f:')
        lines.append('    f.write("spawn\\n")')
    if sleep_s > 0:
        lines.append(f"time.sleep({sleep_s})")
    lines.append('mode = sys.argv[1] if len(sys.argv) > 1 else ""')
    lines.append('if mode == "--json-probe":')
    if bad_json:
        lines.append('    print("not valid json {{{")')
    elif exit_code:
        lines.append(f'    sys.exit({exit_code})')
    else:
        dump = json.dumps(probe if probe is not None else {
            "description": "test tool",
            "timeout_s": 15,
            "cache_ttl_s": 300,
        })
        lines.append(f'    print(json.dumps({dump}))')
    lines.append('elif mode == "--call":')
    if exit_code:
        lines.append(f'    sys.exit({exit_code})')
    elif bad_json:
        lines.append('    print("garbage not json")')
    else:
        lines.append('    print(json.dumps({"ok": True}))')
    lines.append("")
    path.write_text("\n".join(lines))
    if executable:
        path.chmod(0o755)
    else:
        path.chmod(0o644)
    return path


@pytest.mark.asyncio
async def test_discover_tools_finds_executable_and_skips_non_executable(tmp_path: Path) -> None:
    """discover_tools: executable scripts in tools/ + commands from tools.yaml are
    included; non-executable files and wrong extensions are ignored."""
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    write_tool_script(tools_dir, "exec_one.py", probe={"description": "one"})
    # Non-executable sibling must be skipped.
    write_tool_script(tools_dir, "no_exec.py", executable=False)
    # Wrong extension must be skipped even if executable.
    (tools_dir / "readme.txt").write_text("hello")
    os.chmod(tools_dir / "readme.txt", 0o755)
    # YAML-declared command is included.
    yaml_path = tmp_path / "tools.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "yaml_tool": {"command": ["/bin/echo", "hi"]},
        "bad_shape": {"command": "not-a-list"},  # wrong shape -> ignored
    }))

    candidates = discover_tools(str(tmp_path))

    # Expect: exec_one.py + the echo yaml command = 2 candidates
    assert len(candidates) == 2
    # First candidate (sorted by os.listdir) is the script path
    assert candidates[0][0].endswith("exec_one.py")
    # Second is the yaml command vector verbatim
    assert candidates[1] == ["/bin/echo", "hi"]


@pytest.mark.asyncio
async def test_register_accepts_json_probe_and_records_spec_fields(tmp_path: Path) -> None:
    """register: --json-probe stdout JSON object -> spec with parsed fields."""
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    write_tool_script(
        tools_dir,
        "good_tool.py",
        probe={"description": "My tool", "timeout_s": 7.5, "cache_ttl_s": 60},
    )
    candidates = discover_tools(str(tmp_path))
    reg = ToolRegistry()
    tools = await reg.register(candidates)

    assert len(tools) == 1
    spec = tools[0]
    assert spec.name == "good_tool"
    assert spec.description == "My tool"
    assert spec.timeout_s == 7.5
    assert spec.cache_ttl_s == 60.0
    assert reg.rejected == []


@pytest.mark.asyncio
async def test_register_rejects_nonzero_exit(tmp_path: Path) -> None:
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    write_tool_script(tools_dir, "bad_exit.py", exit_code=1)
    reg = ToolRegistry()
    tools = await reg.register(discover_tools(str(tmp_path)))
    assert tools == []
    assert len(reg.rejected) == 1
    assert "returncode_1" in reg.rejected[0][1]


@pytest.mark.asyncio
async def test_register_rejects_bad_json(tmp_path: Path) -> None:
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    write_tool_script(tools_dir, "bad_json.py", bad_json=True)
    reg = ToolRegistry()
    tools = await reg.register(discover_tools(str(tmp_path)))
    assert tools == []
    assert len(reg.rejected) == 1
    assert "parse_error" in reg.rejected[0][1]


@pytest.mark.asyncio
async def test_register_rejects_probe_timeout(tmp_path: Path) -> None:
    """Script sleeping > probe_timeout triggers 'probe_timeout' rejection."""
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    write_tool_script(tools_dir, "slow_tool.py", sleep_s=2.0)
    reg = ToolRegistry()
    t0 = time.monotonic()
    tools = await reg.register(discover_tools(str(tmp_path)), probe_timeout_s=0.2)
    elapsed = time.monotonic() - t0
    assert tools == []
    assert len(reg.rejected) == 1
    assert "probe_timeout" in reg.rejected[0][1]
    # Must return well before the script's own sleep finishes (kill at timeout).
    assert elapsed < 1.5, f"timeout didn't cut it off: elapsed={elapsed}"


@pytest.mark.asyncio
async def test_call_caches_second_call_does_not_spawn(tmp_path: Path) -> None:
    """Two call()s within cache_ttl_s must result in exactly one spawn."""
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    spawn_log = tmp_path / "spawns.log"
    write_tool_script(
        tools_dir,
        "cached_tool.py",
        probe={"description": "c", "timeout_s": 5, "cache_ttl_s": 300},
        side_effect=spawn_log,
    )
    reg = ToolRegistry()
    await reg.register(discover_tools(str(tmp_path)))

    out1 = await reg.call("cached_tool", {"x": 1})
    out2 = await reg.call("cached_tool", {"x": 2})

    assert out1 == {"ok": True}
    assert out2 == {"ok": True}
    # 1 spawn from register (probe) + 1 spawn from first call; second call hits cache.
    spawns = spawn_log.read_text().count("spawn")
    assert spawns == 2


@pytest.mark.asyncio
async def test_call_cache_expiry_respawns(tmp_path: Path) -> None:
    """cache_ttl_s=0.2 + sleep 0.3 -> second call re-spawns the process."""
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    spawn_log = tmp_path / "spawns.log"
    write_tool_script(
        tools_dir,
        "expiring_tool.py",
        probe={"description": "e", "timeout_s": 5, "cache_ttl_s": 0.2},
        side_effect=spawn_log,
    )
    reg = ToolRegistry()
    await reg.register(discover_tools(str(tmp_path)))

    await reg.call("expiring_tool", {})
    await asyncio.sleep(0.3)
    await reg.call("expiring_tool", {})

    # 1 probe + 1 first call + 1 second call (after expiry) = 3
    spawns = spawn_log.read_text().count("spawn")
    assert spawns == 3


@pytest.mark.asyncio
async def test_call_unknown_name_returns_none() -> None:
    reg = ToolRegistry()
    assert await reg.call("does_not_exist", {}) is None