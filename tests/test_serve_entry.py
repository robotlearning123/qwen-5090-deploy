"""The printed serve command is the real user entry (scripts/serve.sh)."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_serve_print_uses_chosen_nvfp4_and_fp8_kv(tmp_path: Path) -> None:
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    fake = venv_bin / "vllm"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    env = os.environ.copy()
    env["QWEN38_VENV"] = str(tmp_path / "venv")
    env["QWEN38_MODEL"] = "unsloth/Qwen3.8-27B-NVFP4"
    env["QWEN38_MAX_MODEL_LEN"] = "16384"

    out = subprocess.check_output(
        [str(ROOT / "scripts" / "serve.sh"), "--print"],
        env=env,
        text=True,
    )
    assert "unsloth/Qwen3.8-27B-NVFP4" in out
    assert "--kv-cache-dtype fp8" in out
    assert "--max-model-len 16384" in out
    assert "--reasoning-parser qwen3" in out
    assert "Qwen3.8-2.4T" not in out
    assert "Qwen3.6" not in out
    # BF16 official must not be the default serve target.
    assert "Qwen/Qwen3.8-27B " not in out + " "


def test_precheck_script_exists_and_is_executable() -> None:
    path = ROOT / "scripts" / "precheck.sh"
    assert path.is_file()
    assert os.access(path, os.X_OK)
    text = path.read_text()
    assert "5090" in text
    assert "BF16" in text
    assert "NVFP4" in text
