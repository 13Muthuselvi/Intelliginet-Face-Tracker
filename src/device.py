"""Pick CUDA when available, otherwise CPU."""

from __future__ import annotations


def resolve_device(preference: str) -> str:
    pref = (preference or "auto").lower()
    if pref not in {"auto", "cpu", "cuda", "0"}:
        return pref
    if pref == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"
