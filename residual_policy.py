from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import PPOResidualConfig


def _build_mlp(input_dim: int, hidden_dim: int, hidden_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Tanh())
        last_dim = hidden_dim
    return nn.Sequential(*layers)


def _make_action_scale(config: PPOResidualConfig) -> torch.Tensor:
    chunk_scales = []
    per_step = torch.tensor(
        [
            config.residual_limit_pos,
            config.residual_limit_pos,
            config.residual_limit_pos,
            config.residual_limit_rot,
            config.residual_limit_rot,
            config.residual_limit_rot,
            config.residual_limit_gripper,
        ],
        dtype=torch.float32,
    )
    for _ in range(config.chunk_size):
        chunk_scales.append(per_step)
    return torch.cat(chunk_scales, dim=0)


class PPOResidualPolicy(nn.Module):
    def __init__(self, config: PPOResidualConfig):
        super().__init__()
        self.config = config
        self.encoder = _build_mlp(config.obs_dim, config.hidden_dim, config.hidden_layers)
        self.actor_mean = nn.Linear(config.hidden_dim, config.flat_action_dim)
        self.critic = nn.Linear(config.hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((config.flat_action_dim,), -2.0, dtype=torch.float32))
        self.register_buffer("action_scale", _make_action_scale(config), persistent=True)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def _distribution(self, obs: torch.Tensor) -> torch.distributions.Independent:
        hidden = self._encode(obs)
        mean = torch.tanh(self.actor_mean(hidden)) * self.action_scale
        std = torch.exp(self.log_std).expand_as(mean)
        base_dist = torch.distributions.Normal(mean, std)
        return torch.distributions.Independent(base_dist, 1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self._encode(obs)
        return self.critic(hidden).squeeze(-1)

    def act(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._distribution(obs)
        action = dist.base_dist.loc if deterministic else dist.rsample()
        action = torch.clamp(action, -self.action_scale, self.action_scale)
        log_prob = dist.log_prob(action)
        value = self.value(obs)
        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._distribution(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.value(obs)
        return log_prob, entropy, value


def build_obs_vector(
    state7: np.ndarray,
    base_action_chunk: np.ndarray,
) -> np.ndarray:
    state7 = np.asarray(state7, dtype=np.float32).reshape(-1)
    base_action_chunk = np.asarray(base_action_chunk, dtype=np.float32)
    return np.concatenate([state7, base_action_chunk.reshape(-1)], axis=0).astype(np.float32)


@dataclass
class PPOResidualOutput:
    corrected_action_chunk: np.ndarray
    residual_action_chunk: np.ndarray


class PPOResidualAdapter:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = PPOResidualConfig.from_dict(checkpoint["config"])
        self.policy = PPOResidualPolicy(self.config)
        self.policy.load_state_dict(checkpoint["model_state"])
        self.policy.to(self.device)
        self.policy.eval()

    def apply(
        self,
        state7: np.ndarray,
        base_action_chunk: np.ndarray,
        *,
        blend: float = 1.0,
    ) -> PPOResidualOutput:
        base_action_chunk = np.asarray(base_action_chunk, dtype=np.float32)
        expected_shape = (self.config.chunk_size, self.config.action_dim)
        if base_action_chunk.shape != expected_shape:
            raise ValueError(
                f"PPO residual expects base action chunk shape {expected_shape}, got {base_action_chunk.shape}"
            )

        obs = build_obs_vector(state7, base_action_chunk)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            residual_flat, _, _ = self.policy.act(obs_tensor, deterministic=True)
        residual = residual_flat.squeeze(0).detach().cpu().numpy().reshape(expected_shape)
        corrected = base_action_chunk + blend * residual
        return PPOResidualOutput(
            corrected_action_chunk=corrected.astype(np.float32),
            residual_action_chunk=residual.astype(np.float32),
        )
