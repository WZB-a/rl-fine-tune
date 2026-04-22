#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from pi05_ppo_finetune.config import PPOResidualConfig
from pi05_ppo_finetune.residual_policy import PPOResidualPolicy
from pi05_ppo_finetune.residual_policy import build_obs_vector


def _iter_rollout_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        file_path for file_path in path.rglob("*") if file_path.suffix.lower() in {".json", ".jsonl"}
    )


def _load_json_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        items: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if "steps" in data and isinstance(data["steps"], list):
        return data["steps"]
    if "episodes" in data and isinstance(data["episodes"], list):
        merged_steps: list[dict[str, Any]] = []
        for episode in data["episodes"]:
            merged_steps.extend(episode.get("steps", []))
        return merged_steps
    raise ValueError(f"Unsupported rollout format: {path}")


def _extract_step_record(step: dict[str, Any]) -> dict[str, Any] | None:
    input_data = step.get("input", step)
    action_data = step.get("action", step)

    state7 = input_data.get("state7_axisangle") or step.get("state7")
    base_action = action_data.get("base_policy_action7_axisangle") or step.get("base_action_chunk")
    final_action = (
        action_data.get("sent_action7_axisangle")
        or action_data.get("post_ppo_action7_axisangle")
        or action_data.get("action7_axisangle")
        or step.get("action_chunk")
    )

    reward = step.get("reward")
    done = step.get("done", False)
    if reward is None and "metrics" in step:
        reward = step["metrics"].get("reward")

    if state7 is None or base_action is None or final_action is None or reward is None:
        return None

    return {
        "state7": np.asarray(state7, dtype=np.float32),
        "base_action": np.asarray(base_action, dtype=np.float32),
        "final_action": np.asarray(final_action, dtype=np.float32),
        "reward": float(reward),
        "done": bool(done),
    }


def load_rollouts(path: Path, config: PPOResidualConfig) -> dict[str, torch.Tensor]:
    obs_list = []
    action_list = []
    reward_list = []
    done_list = []

    for file_path in _iter_rollout_files(path):
        for raw_step in _load_json_file(file_path):
            step = _extract_step_record(raw_step)
            if step is None:
                continue
            if step["base_action"].shape != (config.chunk_size, config.action_dim):
                continue
            if step["final_action"].shape != (config.chunk_size, config.action_dim):
                continue

            residual_action = step["final_action"] - step["base_action"]
            obs_vec = build_obs_vector(step["state7"], step["base_action"])
            obs_list.append(obs_vec)
            action_list.append(residual_action.reshape(-1))
            reward_list.append(step["reward"])
            done_list.append(float(step["done"]))

    if not obs_list:
        raise ValueError("No valid rollout steps found. Each step must contain state7, base action, final action, reward.")

    return {
        "obs": torch.tensor(np.stack(obs_list), dtype=torch.float32),
        "actions": torch.tensor(np.stack(action_list), dtype=torch.float32),
        "rewards": torch.tensor(reward_list, dtype=torch.float32),
        "dones": torch.tensor(done_list, dtype=torch.float32),
    }


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(1, dtype=rewards.dtype, device=rewards.device)

    next_values = torch.cat([values[1:], torch.zeros(1, dtype=values.dtype, device=values.device)])
    for step in reversed(range(rewards.shape[0])):
        mask = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_values[step] * mask - values[step]
        last_adv = delta + gamma * gae_lambda * mask * last_adv
        advantages[step] = last_adv

    returns = advantages + values
    return advantages, returns


def save_checkpoint(
    output_dir: Path,
    policy: PPOResidualPolicy,
    config: PPOResidualConfig,
    *,
    step_count: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "ppo_residual.pt"
    torch.save(
        {
            "config": config.to_dict(),
            "model_state": policy.state_dict(),
            "step_count": step_count,
        },
        checkpoint_path,
    )
    config.save_json(output_dir / "config.json")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO residual head for openpi pi05 inference.")
    parser.add_argument("--rollout_path", type=Path, required=True, help="Rollout json/jsonl file or directory.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the PPO residual checkpoint.")
    parser.add_argument("--init_checkpoint", type=Path, default=None, help="Optional existing PPO residual checkpoint.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--action_dim", type=int, default=7)
    args = parser.parse_args()

    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        config = PPOResidualConfig.from_dict(checkpoint["config"])
    else:
        config = PPOResidualConfig(chunk_size=args.chunk_size, action_dim=args.action_dim)
        checkpoint = None

    device = torch.device(args.device)
    policy = PPOResidualPolicy(config).to(device)
    if checkpoint is not None:
        policy.load_state_dict(checkpoint["model_state"])

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    data = load_rollouts(args.rollout_path, config)
    obs = data["obs"].to(device)
    actions = data["actions"].to(device)
    rewards = data["rewards"].to(device)
    dones = data["dones"].to(device)

    with torch.no_grad():
        old_log_probs, _, values = policy.evaluate_actions(obs, actions)
        advantages, returns = compute_gae(
            rewards,
            dones,
            values,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        if config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    num_samples = obs.shape[0]
    minibatch_size = min(config.minibatch_size, num_samples)
    total_updates = 0

    for _ in range(config.ppo_epochs):
        permutation = torch.randperm(num_samples, device=device)
        for start in range(0, num_samples, minibatch_size):
            total_updates += 1
            batch_idx = permutation[start : start + minibatch_size]

            batch_obs = obs[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]

            log_probs, entropy, values_pred = policy.evaluate_actions(batch_obs, batch_actions)
            ratio = torch.exp(log_probs - batch_old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio)

            policy_loss_1 = ratio * batch_advantages
            policy_loss_2 = clipped_ratio * batch_advantages
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
            value_loss = F.mse_loss(values_pred, batch_returns)
            entropy_loss = entropy.mean()

            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

    checkpoint_path = save_checkpoint(args.output_dir, policy, config, step_count=total_updates)
    print(f"Saved PPO residual checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
