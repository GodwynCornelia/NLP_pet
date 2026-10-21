import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_resource_lock(config):
    import json

    path = ROOT / config.get("resources", {}).get("lock_file", "configs/resources.lock.json")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock["schema_version"] != 1 or lock["encoder_id"] != config["model"]["encoder_id"]:
        raise ValueError("resource_lock_incompatible")
    return lock


def load_config(path=None):
    path = Path(path or os.getenv("SUPPORT_ROUTER_CONFIG", ROOT / "configs/default.toml"))
    with path.open("rb") as f:
        config = tomllib.load(f)
    for key, env in [("artifact_dir", "ARTIFACT_DIR"), ("reports_dir", "REPORTS_DIR")]:
        if os.getenv(env):
            config["project"][key] = os.environ[env]
    for key in ("data_dir", "artifact_dir", "reports_dir"):
        config["project"][key] = str((ROOT / config["project"][key]).resolve())
    return config


def portable_config(config):
    return {k: v for k, v in config.items() if k != "project"} | {
        "project": {"seed": config["project"]["seed"]}
    }
