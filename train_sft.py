# train_sft.py - 16bit快速跑通版（无量化/PEFT，适配RTX 4090D）
import os
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)

# ================== 极简配置 ==================
MODEL_PATH = os.environ.get("MODEL_PATH", "/root/autodl-tmp/Qwen2.5-Coder-1.5B")
OUTPUT_DIR = os.environ.get("SFT_OUTPUT", "./sft_model")
TRAIN_FILE = os.environ.get("SFT_DATA", "./data/sft_data.jsonl")

# 训练配置（极简，确保跑通）
BATCH_SIZE = 1                  # 单样本批次（显存占用最低）
GRAD_ACCUM = 2                  # 梯度累积（总批次=2，加速训练）
EPOCHS = 1                      # 只跑1轮（快速验证）
MAX_LEN = 512                   # 缩短序列长度（省显存）
LR = 2e-5

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== 数据预处理 ==================
def prepare_lines(example):
    """简化格式，避免复杂处理"""
    return {
        "text": f"<s>[INST] {example['prompt'].strip()} [/INST] {example['response'].strip()} </s>"
    }

def main():
    # 1. 加载tokenizer（极简配置）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token  # 补全pad_token
    tokenizer.padding_side = "right"

    # 2. 加载模型（16bit精度，无量化）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",  # 自动分配GPU
        # dtype=torch.bfloat16,  # <-- 这是旧的，错误的行，应该删除或注释掉
        torch_dtype=torch.bfloat16,  # <-- 这是新的，正确的行
        trust_remote_code=True
    )
    
    model.config.use_cache = False  # 训练禁用缓存（省显存）

    # 3. 加载数据（简化处理）
    try:
        ds = load_dataset("json", data_files=TRAIN_FILE, split="train")
        print(f"加载数据集：{len(ds)} 条样本")
        ds = ds.map(prepare_lines)  # 仅格式化文本，不额外过滤
    except Exception as e:
        print(f"数据加载失败：{e}")
        return

    # 4. 分词（极简版）
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length"
        )
    tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    # 5. 数据整理器（因果语言模型专用）
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 6. 训练参数（只保留必要项，确保跑通）
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        
        # 16bit训练（核心）
        bf16=True,
        fp16=False,
        
        # 省显存关键配置
        gradient_checkpointing=True,
        
        # 禁用不必要功能
        save_strategy="no",  # 不保存检查点（快速跑通）
        logging_steps=1,     # 每步打印日志（方便调试）
        report_to="none",    # 禁用外部日志工具
        remove_unused_columns=False
    )

    # 7. 初始化Trainer（原生版，无自定义）
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator
    )

    # 8. 开始训练（快速验证）
    print("开始训练...")
    try:
        trainer.train()
        print("✅ 训练成功跑通！")
    except Exception as e:
        print(f"❌ 训练报错：{e}")
        return

    # 9. 可选：保存最终模型（如果需要）
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"模型保存到：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()