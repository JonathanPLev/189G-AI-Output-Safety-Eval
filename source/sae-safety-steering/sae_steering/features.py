from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class FeatureBank:
    harmful_indices:   list[int] = field(default_factory=list)
    unharmful_indices: list[int] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "FeatureBank":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(
            harmful_indices=data.get("harmful_indices", []),
            unharmful_indices=data.get("unharmful_indices", []),
        )

    def to_yaml(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                {
                    "harmful_indices":   self.harmful_indices,
                    "unharmful_indices": self.unharmful_indices,
                },
                f,
            )
        print(f"Feature bank saved → {path}")

    def __repr__(self) -> str:
        return (
            f"FeatureBank("
            f"harmful={len(self.harmful_indices)}, "
            f"unharmful={len(self.unharmful_indices)})"
        )
