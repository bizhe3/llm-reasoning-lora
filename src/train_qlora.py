import os
from typing import Dict, Any
from pathlib import Path
import sys

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

# ========= 1. 统一设置项目根目录（本地 / Colab 都适用） =========
# 当前文件: <repo_root>/src/train_qlora.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 上上级目录 = 仓库根目录
sys.path.append(str(PROJECT_ROOT))  # 让 Python 能够找到 configs 包

from configs import qlora_config as cfg  # noqa: E402

DATA_PATH = PROJECT_ROOT / cfg.DATA_PATH          # e.g. data/processed/merged_instructions.jsonl
OUTPUT_DIR = PROJECT_ROOT / cfg.OUTPUT_DIR        # e.g. outputs/qlora-qwen2.5-7b-sft


def load_merged_dataset():
    """从 jsonl 加载 merged 数据集"""
    dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")

    # 可选：只取前 N 条做 sanity run
    if cfg.MAX_TRAIN_SAMPLES is not None:
        dataset = dataset.select(range(cfg.MAX_TRAIN_SAMPLES))
        print(f"Using subset of dataset: {len(dataset)} samples")
    else:
        print(f"Using full dataset: {len(dataset)} samples")

    return dataset


def format_example(example: Dict[str, Any], tokenizer) -> str:
    """
    把我们自己的 schema 转成一个字符串 prompt。
    对 Qwen2.5-Instruct，用 chat template 会更贴合它的风格。
    """
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

    # SFTTrainer 会负责 tokenize，这里只返回纯字符串
    return tokenizer.apply_chat_template(messages, tokenize=False)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("Loading dataset from:", DATA_PATH)

    dataset = load_merged_dataset()

    print("Loading tokenizer and base model:", cfg.BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(cfg.BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # QLoRA: 4-bit 量化配置（Colab 一般用 float16 更稳）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,   # T4 / L4 上用 fp16 更通用
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # LoRA 配置
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

    # ===== 精度设置：bf16 优先，其次 fp16（只能开一个） =====
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    print(f"use_fp16 = {use_fp16}, use_bf16 = {use_bf16}")

    # ===== TRL 的 SFTConfig（相当于新版 TrainingArguments） =====
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=cfg.NUM_EPOCHS,
        per_device_train_batch_size=cfg.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRAD_ACCUM_STEPS,
        learning_rate=cfg.LEARNING_RATE,
        logging_steps=cfg.LOGGING_STEPS,
        save_strategy=cfg.SAVE_STRATEGY,
        max_length=cfg.MAX_SEQ_LENGTH,  # 新版在 SFTConfig 里用 max_length 控制截断
        fp16=use_fp16,
        bf16=use_bf16,
        report_to=[],                   # 不用 wandb 等
        # 你也可以在这里加 packing / assistant_only_loss 等 TRL 新参数
        # packing=False,
    )

    # ===== 新版 TRL：formatting_func 接收单条 example，返回字符串 =====
    def formatting_func(example: Dict[str, Any]) -> str:
        return format_example(example, tokenizer)

    # ===== 新版 TRL：用 processing_class 代替 tokenizer =====
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
        processing_class=tokenizer,  # 0.12+ 起 tokenizer 参数更名为 processing_class
    )

    print("Start training...")
    trainer.train()
    print("Training finished. Saving model to:", OUTPUT_DIR)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
