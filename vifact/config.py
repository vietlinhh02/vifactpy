from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class ViFactConfig:
    system: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    loader: Dict[str, Any] = field(default_factory=dict)
    preprocess: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    retrieval: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ViFactConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def update(self, overrides: Dict[str, Any] | None = None) -> None:
        if not overrides:
            return
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(getattr(self, k, {}), dict):
                getattr(self, k).update(v)
            else:
                setattr(self, k, v)

    # Convenience accessors
    @property
    def raw_dir(self) -> Path:
        return Path(self.data.get("raw_dir", "data/raw")).resolve()

    @property
    def processed_dir(self) -> Path:
        return Path(self.data.get("processed_dir", "data/processed")).resolve()

    @property
    def sample_dir(self) -> Path:
        return Path(self.data.get("sample_dir", "data/sample")).resolve()
