"""Safe pickle helpers for loading and saving artifacts.

These utilities aim to reduce accidental unsafe deserialization by
constraining where pickles can be read from and by providing a simple
atomic write helper for outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union
import pickle
import os


def _ensure_within_root(path: Path, allowed_root: Optional[Path]) -> None:
    if allowed_root is None:
        return
    resolved = path.resolve()
    root = allowed_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"Refusing to access file outside allowed root: {resolved} (root: {root})")


def safe_pickle_load(path: Union[str, Path], *, allowed_root: Optional[Union[str, Path]] = None) -> Any:
    """Load a pickle file with basic safety constraints.

    Args:
        path: Path to the pickle file.
        allowed_root: Optional directory; if provided, the file must be inside it.

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the file is outside the allowed_root or not a pickle extension.
        FileNotFoundError: If the path does not exist.
        pickle.UnpicklingError: If unpickling fails.
    """
    p = Path(path)
    if p.suffix not in {".pkl", ".pickle"}:
        raise ValueError(f"Refusing to load non-pickle file: {p}")
    if allowed_root is not None:
        _ensure_within_root(p, Path(allowed_root))
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    with open(p, "rb") as f:
        return pickle.load(f)


def safe_pickle_dump(obj: Any, path: Union[str, Path], *, temp_suffix: str = ".tmp") -> Path:
    """Atomically write a pickle file to disk.

    Writes to a temporary filename in the same directory, then renames
    to the final path to avoid partial writes.

    Args:
        obj: Python object to serialize.
        path: Destination file path.
        temp_suffix: Suffix to use for the temporary file.

    Returns:
        Path to the final written file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + temp_suffix)
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, p)
    return p

