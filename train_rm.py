# train_rm.py - 修复全局变量声明顺序问题 和 dtype 参数错误
import os
import json
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
import torch

# 导入评估函数
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# 配置参数
MODEL_PATH = os.environ.get("MODEL_PATH", "/root/autodl-tmp/Qwen2.5-Coder-1.5B")
TRAIN_FILE = os.environ.get("RM_DATA", "./data/rm_data.jsonl")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
EPOCHS = int(os.environ.get("EPOCHS", "3"))

# 全局输出目录变量
OUTPUT_DIR = "/root/autodl-tmp/sft/rm_model"  # 默认输出目录

def prepare_texts(example):
    return {
        "text": f"<s>[INST] {example['prompt'].strip()} [/INST] {example['response'].strip()} </s>",
        "labels": example["score"]
    }

def main():
    # 首先声明全局变量
    global OUTPUT_DIR
    
    # 检查并创建输出目录
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # 测试写入权限
        test_file = os.path.join(OUTPUT_DIR, "test_write.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print(f"使用输出目录: {OUTPUT_DIR}")
    except Exception as e:
        print(f"错误: 无法在 {OUTPUT_DIR} 写入文件 - {str(e)}")
        print("尝试使用/tmp目录作为最后的备选方案")
        # 修改全局变量
        OUTPUT_DIR = "/tmp/rm_model"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"已切换到备选目录: {OUTPUT_DIR}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 加载模型 - 修正：使用 torch_dtype 而非 dtype
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=1,
        device_map="auto",
        trust_remote_code=True,
        # dtype=torch.bfloat16, # <-- 旧的，错误的参数名
        torch_dtype=torch.bfloat16, # <-- 新的，正确的参数名
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.problem_type = "regression"

    # 加载并预处理数据
    ds = load_dataset("json", data_files=TRAIN_FILE, split="train")
    ds = ds.map(prepare_texts)
    ds = ds.train_test_split(test_size=0.1)
    
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            padding="max_length"
        )
        result["labels"] = examples["labels"]
        return result
    
    tokenized_datasets = ds.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # 数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 自定义Trainer处理损失计算
    class RegressionTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if "labels" not in inputs:
                raise ValueError("输入数据中没有找到'labels'键")
                
            labels = inputs.pop("labels")
            outputs = model(**inputs) # 修复可能的多余空格
            logits = outputs.logits
            loss = torch.nn.functional.mse_loss(logits.squeeze(), labels.float())
            return (loss, outputs) if return_outputs else loss

    # 训练参数
    training_args = TrainingArguments(
        output_dir="/tmp/rm_checkpoints",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        bf16=True,
        eval_strategy="epoch", # 注意：较新版本transformers使用 eval_strategy 替代 evaluation_strategy
        save_strategy="no",
        logging_steps=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # 如果升级 accelerate 后仍有 dispatch_batches 错误，可以尝试显式添加或移除这个参数
        # dispatch_batches=None, # 或者尝试 dispatch_batches=False
    )

    # 评估指标
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.reshape(-1)
        labels = labels.reshape(-1)
        
        mse = mean_squared_error(labels, predictions)
        pearson_corr = pearsonr(labels, predictions)[0]
        
        return {"mse": mse, "pearson": pearson_corr}

    # 初始化Trainer并训练
    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 开始训练
    trainer.train()
    
    # 保存模型
    model.save_pretrained(OUTPUT_DIR, safe_serialization=False)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 保存配置文件
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model.config.to_dict(), f, indent=2, ensure_ascii=False)
        
    print(f"模型已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()