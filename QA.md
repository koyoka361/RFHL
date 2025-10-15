
### 常见问题

#### Q1: 显存不足怎么办？
- 减小 `BATCH_SIZE` 到 1
- 降低 `MAX_LEN` 到 256
- 启用梯度检查点

#### Q2: 训练速度太慢？
- 增加 `GRAD_ACCUM` 值
- 使用混合精度训练
- 减少训练轮数

#### Q3: 模型质量不佳？
- 检查训练数据质量
- 增加训练轮数
- 调整学习率

### 高级配置

#### 自定义训练参数
编辑训练脚本中的参数：
```python
# train_sft.py
BATCH_SIZE = 2                  # 批次大小
GRAD_ACCUM = 4                  # 梯度累积
EPOCHS = 2                      # 训练轮数
MAX_LEN = 1024                  # 最大长度
LR = 1e-5                       # 学习率
```

#### 多GPU训练
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch train_sft.py
```

### 模型评估

#### 自动评估
训练脚本会自动计算：
- 困惑度 (Perplexity)
- 准确率 (Accuracy)
- F1分数

#### 人工评估
可以通过以下方式测试模型：
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./sft_model")
tokenizer = AutoTokenizer.from_pretrained("./sft_model")

prompt = "用Python实现快速排序算法"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
print(tokenizer.decode(outputs[0]))
```

### 部署使用

#### 模型加载
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载训练好的模型
model_path = "./ppo_model"  # 或 "./sft_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

#### 代码生成
```python
def generate_code(prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        do_sample=True,
        top_p=0.95
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 使用示例
code = generate_code("用Python实现二叉树遍历")
print(code)
```

### 最佳实践

1. **数据质量**: 确保训练数据准确、多样
2. **逐步训练**: 先SFT再RM最后PPO
3. **监控指标**: 密切关注训练损失和评估指标
4. **保存检查点**: 定期保存模型状态
5. **环境隔离**: 使用虚拟环境避免依赖冲突

### 技术支持

如遇到问题，请检查：
- CUDA和PyTorch版本兼容性
- 模型文件完整性
- 数据格式正确性
- 显存使用情况

---

祝你训练顺利！🚀
