from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class PPOResidualConfig:
    state_dim: int = 7
    action_dim: int = 7
    chunk_size: int = 8
    hidden_dim: int = 256
    hidden_layers: int = 2

    residual_limit_pos: float = 0.015
    residual_limit_rot: float = 0.08
    residual_limit_gripper: float = 0.20

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    ppo_epochs: int = 10
    minibatch_size: int = 64
    normalize_advantages: bool = True

    @property
    def flat_action_dim(self) -> int:
        return self.action_dim * self.chunk_size

    @property
    def obs_dim(self) -> int:
        return self.state_dim + self.flat_action_dim

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PPOResidualConfig":
        valid_keys = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in valid_keys})

    @classmethod
    def from_json(cls, path: str | Path) -> "PPOResidualConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
