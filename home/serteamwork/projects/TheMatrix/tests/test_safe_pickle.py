"""Tests for scripts.safe_pickle helpers."""

from pathlib import Path
import pickle

import pytest

from scripts.safe_pickle import safe_pickle_load, safe_pickle_dump


def test_safe_pickle_dump_and_load(tmp_path: Path) -> None:
    data = {"k": "v"}
    dest = tmp_path / "model.pkl"

    written = safe_pickle_dump(data, dest)
    assert written == dest
    assert dest.exists()

    loaded = safe_pickle_load(dest, allowed_root=tmp_path)
    assert loaded == data


def test_safe_pickle_load_rejects_non_pickle(tmp_path: Path) -> None:
    txt = tmp_path / "data.txt"
    txt.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError):
        safe_pickle_load(txt, allowed_root=tmp_path)


def test_safe_pickle_load_rejects_outside_root(tmp_path: Path) -> None:
    other_dir = tmp_path.parent / f"{tmp_path.name}_other"
    other_dir.mkdir(exist_ok=True)
    outside = other_dir / "model.pkl"
    with open(outside, "wb") as f:
        pickle.dump({"k": 1}, f)

    with pytest.raises(ValueError):
        safe_pickle_load(outside, allowed_root=tmp_path)
