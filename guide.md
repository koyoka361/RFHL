# RLHF 代码生成模型训练指南

## 快速开始

本指南将帮助你快速上手RLHF代码生成模型的训练流程。

### 环境准备

#### 1. 系统要求
- Python 3.8+
- CUDA 11.8+
- GPU显存 ≥ 16GB (推荐24GB)

#### 2. 安装依赖
```bash
pip install transformers datasets torch scipy scikit-learn accelerate
```

#### 3. 模型下载
```bash
# 下载基础模型 (Qwen2.5-Coder-1.5B)
git clone https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B /root/autodl-tmp/Qwen2.5-Coder-1.5B
```

### 数据准备

#### 数据格式说明

**SFT训练数据** (`sft_data.jsonl`):
```json
{"prompt": "用Python实现二分查找算法", "response": "def binary_search(arr, target):\n    ..."}
```

**RM训练数据** (`rm_data.jsonl`):
```json
{"prompt": "用Python实现二分查找算法", "response": "def binary_search(...)", "score": 0.9}
```

**PPO提示词** (`ppo_prompts.jsonl`):
```json
{"prompt": "用Python实现二分查找算法"}
```

### 训练流程

#### 步骤1: SFT有监督微调

```bash
# 设置环境变量
export MODEL_PATH="/root/autodl-tmp/Qwen2.5-Coder-1.5B"
export SFT_DATA="./sft_data.jsonl"
export BATCH_SIZE=1
export EPOCHS=1

# 开始训练
python train_sft.py
```

**训练参数说明**:
- `BATCH_SIZE`: 1 (最小显存占用)
- `MAX_LEN`: 512 (序列长度)
- `LR`: 2e-5 (学习率)
- `EPOCHS`: 1 (训练轮数)

#### 步骤2: 奖励模型训练

```bash
# 设置环境变量
export MODEL_PATH="/root/autodl-tmp/Qwen2.5-Coder-1.5B"
export RM_DATA="./rm_data.jsonl"
export BATCH_SIZE=8
export EPOCHS=3

# 开始训练
python train_rm.py
```

#### 步骤3: 一键运行完整流程

```bash
# 运行所有训练步骤
bash run_all.sh
```

### 训练监控

#### 日志输出
训练过程中会显示：
- 训练进度
- 损失值变化
- 评估指标
- 预计完成时间

#### 输出目录
