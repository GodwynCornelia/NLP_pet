import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

PREPROCESSING = "nfc-strip-whitespace-v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temp.replace(path)


def file_info(path):
    return {"sha256": sha(path), "bytes": Path(path).stat().st_size}


def check_file(path, info):
    if not Path(path).is_file() or file_info(path) != info:
        raise ValueError(f"artifact_hash_mismatch: {Path(path).name}")


def versions():
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "sentence-transformers",
        "huggingface-hub",
        "fastapi",
        "pydantic",
        "streamlit",
        "joblib",
    ]
    return {p: importlib.metadata.version(p) for p in packages}


def hardware():
    import subprocess

    cpu = platform.processor()
    if os.name != "nt" and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    if os.name == "nt":
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True,
            text=True,
        )
        cpu = result.stdout.strip() or cpu
    return {
        "cpu": cpu,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "available_ram_bytes": psutil.virtual_memory().available,
        "process_rss_bytes": psutil.Process().memory_info().rss,
    }


def setup_compute(config):
    import torch
    from threadpoolctl import threadpool_limits

    threads = config["model"]["threads"]
    torch.set_num_threads(threads)
    np.random.seed(config["project"]["seed"])
    torch.manual_seed(config["project"]["seed"])
    # Keep controller alive; all training and runtime linear algebra shares this limit.
    return threadpool_limits(limits=threads)


def run_paths(config, run_id=None):
    root = Path(config["project"]["artifact_dir"])
    run_id = run_id or read_json(root / "current.json")["run_id"]
    if (
        not isinstance(run_id, str)
        or not run_id
        or Path(run_id).name != run_id
        or run_id in {".", ".."}
    ):
        raise ValueError("invalid_run_id")
    artifact = root / run_id
    report = Path(config["project"]["reports_dir"]) / run_id
    return run_id, artifact, report
