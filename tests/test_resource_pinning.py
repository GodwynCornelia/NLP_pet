from pathlib import Path

import pandas as pd
import pytest

from support_router.common import file_info
from support_router.config import load_config
from support_router.data import download


@pytest.fixture
def pinned_download(tmp_path, monkeypatch):
    import support_router.data as data

    revision = "a" * 40
    encoder_revision = "b" * 40
    content = {}
    for name, count in [("train", 10003), ("test", 3080)]:
        content[name] = (
            pd.DataFrame(
                {
                    "text": [f"text {i}" for i in range(count)],
                    "category": [f"class-{i % 77}" for i in range(count)],
                }
            )
            .to_csv(index=False)
            .encode()
        )
    encoder_bytes = b"unit-test-only-not-real-weights"
    expected = {}
    for name, body in content.items():
        path = tmp_path / f"expected-{name}"
        path.write_bytes(body)
        expected[f"raw/{revision}/{name}.csv"] = file_info(path)
    encoder_file = tmp_path / "expected-model"
    encoder_file.write_bytes(encoder_bytes)
    expected[f"resources/{encoder_revision}/model.safetensors"] = file_info(encoder_file)
    config = load_config()
    config["project"]["data_dir"] = str(tmp_path / ".cache" / "clean-data")
    lock = {
        "schema_version": 1,
        "dataset_commit": revision,
        "encoder_id": config["model"]["encoder_id"],
        "encoder_revision": encoder_revision,
        "files": expected,
    }
    monkeypatch.setattr(data, "load_resource_lock", lambda config: lock)
    requests = []

    class Response:
        pass

    def get(url):
        assert revision in url and "master" not in url and "api.github" not in url
        requests.append(url)
        result = Response()
        result.content = content[Path(url).stem]
        return result

    def snapshot(repo_id, **kwargs):
        assert kwargs["revision"] == encoder_revision
        assert kwargs["token"] is False
        assert kwargs["allow_patterns"] == ["model.safetensors"]
        path = Path(kwargs["local_dir"])
        path.mkdir(parents=True)
        (path / "model.safetensors").write_bytes(encoder_bytes)

    monkeypatch.setattr(data, "request", get)
    monkeypatch.setattr(data, "snapshot_download", snapshot)
    return config, content, requests


def test_clean_download_uses_pinned_revisions_and_checks_cache(pinned_download):
    config, _, requests = pinned_download
    source = download(config)
    assert len(source["files"]) == 3  # Parent .cache must not hide the model from its manifest.
    assert source["commit_pinned"] and len(requests) == 2
    assert download(config) == source and len(requests) == 2
    path = Path(config["project"]["data_dir"]) / source["datasets"]["train"]["path"]
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash_mismatch"):
        download(config)
    assert path.read_bytes() == b"tampered"


def test_wrong_remote_bytes_are_not_promoted(pinned_download):
    config, content, _ = pinned_download
    content["train"] = content["train"].replace(b"text 0", b"text x", 1)
    with pytest.raises(ValueError, match="hash_mismatch"):
        download(config)
    assert not list(Path(config["project"]["data_dir"]).rglob("train.csv"))
