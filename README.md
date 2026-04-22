# pi05 PPO Finetune
- PPO 部分可以单独训练、单独开关
## 文件

- `config.py`: PPO residual 超参数
- `residual_policy.py`: residual actor-critic 和推理适配器
- `train_ppo.py`: 用 rollout json/jsonl 训练 residual

## 训练数据格式

`train_ppo.py` 支持读取目录下的 `json/jsonl`，每一步至少要能提取出：

- `input.state7_axisangle`
- `action.base_policy_action7_axisangle`
- `action.sent_action7_axisangle`
- `reward`
- `done`

如果你直接使用改过的 `inference_server_openpi_pi05.py` 记录日志，它会多写：

- `base_policy_action7_axisangle`
- `post_ppo_action7_axisangle`
- `ppo_residual7_axisangle`
- `sent_action7_axisangle`

其中：

- `base_policy_action7_axisangle` 是 openpi 原始输出
- `post_ppo_action7_axisangle` 是 PPO 修正后的输出
- `sent_action7_axisangle` 是最终发给机器人前、经过放大/安全限制后的输出

做 PPO 训练时，优先使用没有安全裁剪前的 `post_ppo_action7_axisangle`。

## 训练示例

```bash
cd /data3/yinmenghao/code/openpi

python realworld_deploy/server/pi05_ppo_finetune/train_ppo.py \
  --rollout_path realworld_deploy/server/log/my_rewarded_rollouts.json \
  --output_dir realworld_deploy/server/pi05_ppo_finetune/checkpoints/run_001
```

输出 checkpoint:

- `ppo_residual.pt`
- `config.json`

## 推理接入

启动推理服务器时加：

```bash
python realworld_deploy/server/inference_server_openpi_pi05.py \
  --checkpoint_dir /path/to/openpi/checkpoint \
  --config_name pi05_assembly_things_lora \
  --ppo_checkpoint realworld_deploy/server/pi05_ppo_finetune/checkpoints/run_001/ppo_residual.pt \
  --ppo_blend 1.0
```

也可以不用命令行，改环境变量：

```bash
export OPENPI_PI05_PPO_CHECKPOINT=/path/to/ppo_residual.pt
export OPENPI_PI05_PPO_BLEND=0.7
```

## 说明

这里参考了仓库里现有 PI RL/PPO 的裁剪目标思路，但没有直接照搬离散 token policy 的 trainer。
原因是 `openpi pi05` 当前主干是连续动作 flow-matching 推理，不是离散 action token policy。
