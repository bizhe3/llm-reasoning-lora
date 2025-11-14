import os
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


# ========== 这里是内置配置，不再依赖 configs 包 ==========

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 基于当前文件自动推项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "merged_instructions.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "lora-qwen2.5-7b-sft")

# sanity run：先小规模跑通
MAX_TRAIN_SAMPLES = 2000    # None 表示用全部数据
MAX_SEQ_LENGTH = 1024       # 显存不够可以改成 768 或 512

PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8

NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
LOGGING_STEPS = 50
SAVE_STRATEGY = "epoch"


# ========== 下面开始是真正的训练逻辑 ==========

def load_merged_dataset():
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    if MAX_TRAIN_SAMPLES is not None:
        dataset = dataset.select(range(MAX_TRAIN_SAMPLES))
        print(f"Using subset of dataset: {len(dataset)} samples")
    else:
        print(f"Using full dataset: {len(dataset)} samples")
    return dataset


def build_prompt(example: Dict[str, Any], tokenizer) -> str:
    """把一条样本转成完整的对话 prompt 文本"""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")

    if inp:
        user_content = f"{instruction}\n\n附加信息：{inp}"
    else:
        user_content = instruction

    messages = [
        {"role": "system", "content": "You are a helpful assistant good at step-by-step reasoning."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("DATA_PATH:", DATA_PATH)
    print("OUTPUT_DIR:", OUTPUT_DIR)

    print("Loading dataset...")
    raw_dataset = load_merged_dataset()

    print("Loading tokenizer and base model:", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # 省显存
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # 配置 LoRA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 预处理：把文本转成 token，并设置 labels = input_ids（自回归 LM）
    def preprocess(example):
        text = build_prompt(example, tokenizer)
        tokenized = tokenizer(
            text,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("Tokenizing dataset...")
    tokenized_dataset = raw_dataset.map(
        preprocess,
        remove_columns=raw_dataset.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        save_strategy=SAVE_STRATEGY,
        fp16=torch.cuda.is_available(),
        bf16=False,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",  # 对 LoRA 友好
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Start training (LoRA + HF Trainer)...")
    trainer.train()
    print("Training finished. Saving model to:", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
