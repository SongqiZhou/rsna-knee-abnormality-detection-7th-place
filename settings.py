"""The paths every script reads and writes, from SETTINGS.json at the repository root (the only place they are set).
Relative paths are relative to the repository root."""
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
_S = json.load(open(os.path.join(ROOT, "SETTINGS.json")))


def path(key, *parts):
    """SETTINGS.json[key] as an absolute path, joined with parts."""
    p = _S[key] if os.path.isabs(_S[key]) else os.path.join(ROOT, _S[key])
    return os.path.join(p, *parts)
