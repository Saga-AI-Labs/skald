"""Tests for ``identity`` — the single checkpoint-hashing entry point.

Covers file hashing, deterministic directory hashing (order- and
metadata-independent), the ``hash_model`` dispatcher, and the EXL3-relevant
property that identity follows stored bytes (re-quantized bytes hash
differently — that is the point).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from identity import hash_checkpoint, hash_dir, hash_model


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_file_hash_is_content_sha256(tmp_path):
    f = tmp_path / "w.bin"
    f.write_bytes(b"weights")
    assert hash_checkpoint(f) == _sha(b"weights")


def test_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        hash_checkpoint("/no/such/file.bin")
    with pytest.raises(FileNotFoundError):
        hash_dir("/no/such/dir")
    with pytest.raises(FileNotFoundError):
        hash_model("/no/such/artifact")


def test_dir_hash_is_order_deterministic(tmp_path):
    (tmp_path / "b").write_bytes(b"2")
    (tmp_path / "a").write_bytes(b"1")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c").write_bytes(b"3")
    first = hash_dir(tmp_path)
    # rebuild in a different creation order; same bytes, same hash
    import shutil

    other = tmp_path / "other"
    (other / "sub").mkdir(parents=True)
    (other / "sub" / "c").write_bytes(b"3")
    (other / "a").write_bytes(b"1")
    (other / "b").write_bytes(b"2")
    assert hash_dir(other) == first
    shutil.rmtree(other)
    # content change changes identity
    (tmp_path / "a").write_bytes(b"1!")
    assert hash_dir(tmp_path) != first


def test_dir_hash_ignores_hf_fetch_metadata_by_default(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"w")
    cache = tmp_path / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "bookkeeping.json").write_bytes(b"fetched-at-some-time")
    without_meta = hash_dir(tmp_path)
    (cache / "more.json").write_bytes(b"changes-on-redownload")
    assert hash_dir(tmp_path) == without_meta
    assert hash_dir(tmp_path, exclude=None) != without_meta


def test_hash_model_dispatches(tmp_path):
    f = tmp_path / "w.bin"
    f.write_bytes(b"weights")
    assert hash_model(f) == hash_checkpoint(f)
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "w").write_bytes(b"weights")
    assert hash_model(d) == hash_dir(d)


def test_adapters_agree_with_identity(tmp_path):
    """Every adapter's helper must equal the shared module's answer."""
    from adapters.saga import _checkpoint_sha256 as saga_hash
    from adapters.jlens import _checkpoint_sha256 as jlens_hash
    from adapters.pi50 import _checkpoint_sha256 as pi50_hash

    f = tmp_path / "w.bin"
    f.write_bytes(b"weights")
    assert saga_hash(f) == pi50_hash(f) == hash_model(f, exclude=None)
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "w").write_bytes(b"weights")
    assert saga_hash(d) == pi50_hash(d) == hash_model(d, exclude=None)
    assert jlens_hash(d) == hash_model(d)
