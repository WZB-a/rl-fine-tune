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

PPO 训练时，优先使用 `post_ppo_action7_axisangle`。

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

启动推理服务器：

```bash
python realworld_deploy/server/inference_server_openpi_pi05.py \
  --checkpoint_dir /path/to/openpi/checkpoint \
  --config_name pi05_assembly_things_lora \
  --ppo_checkpoint realworld_deploy/server/pi05_ppo_finetune/checkpoints/run_001/ppo_residual.pt \
  --ppo_blend 1.0
```


