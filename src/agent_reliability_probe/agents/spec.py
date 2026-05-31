"""Model-spec parsing: attach an optional thinking level to a provider model id.

    gemini/gemini-3.5-flash:medium  -> ("gemini/gemini-3.5-flash", "medium")
    moonshot/kimi-k2.6              -> ("moonshot/kimi-k2.6", None)

Safe because provider model ids use "/", "-", and "." — never ":" — and the
suffix is only treated as a thinking level when it's one of the known values.
"""

from __future__ import annotations

_EFFORTS = {"minimal", "low", "medium", "high", "none"}


def parse_model_spec(spec: str) -> tuple[str, str | None]:
    model, sep, effort = spec.rpartition(":")
    if sep and effort.lower() in _EFFORTS:
        return model, effort.lower()
    return spec, None
