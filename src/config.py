from pathlib import Path
from typing import Optional
import yaml

_DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "example.yaml"


def load(path: Optional[Path] = None) -> dict:
    target = path or _DEFAULT_CONFIG
    with open(target, "r") as f:
        return yaml.safe_load(f)


def audio_cfg(cfg: dict) -> dict:
    return cfg.get("audio", {})


def monitor_cfg(cfg: dict) -> dict:
    return cfg.get("monitor", {})


def bpm_cfg(cfg: dict) -> dict:
    return cfg.get("bpm", {})


def link_cfg(cfg: dict) -> dict:
    return cfg.get("link", {})
