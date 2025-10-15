#!/bin/bash
# 一键运行Qwen2.5-Coder-1.5B的SFT→RM→PPO训练流程

# 确保目录存在
mkdir -p data ./sft_model ./rm_model ./ppo_model

# 安装依赖
echo "安装依赖包..."
pip install -r requirements.txt

# 训练SFT模型
echo "开始训练SFT模型..."
export MODEL_PATH="/root/autodl-tmp/Qwen2.5-Coder-1.5B"
export BATCH_SIZE=4
export EPOCHS=3
export LOAD_IN_8BIT=False  # 如显存足够(>24GB)，可改为False
python train_sft.py

# 训练奖励模型
echo "开始训练奖励模型..."
export MODEL_PATH="/root/autodl-tmp/Qwen2.5-Coder-1.5B"
export BATCH_SIZE=8
export EPOCHS=3
export LOAD_IN_8BIT=False
python train_rm.py

# 运行PPO训练
echo "开始PPO训练..."
export BASE_MODEL="./sft_model"
export RM_MODEL="./rm_model"
export BATCH_SIZE=2
export PPO_EPOCHS=5
export LOAD_IN_8BIT=False
python train_ppo.py

echo "所有训练流程已完成！最终模型保存在 ./ppo_model 目录"
    