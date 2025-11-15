# configs/qlora_config.py
"""
QLoRA 训练超参配置文件
会被 src/train_qlora.py 中的：

    from configs import qlora_config as cfg

引用，然后通过 cfg.DATA_PATH, cfg.OUTPUT_DIR 等字段访问。
"""

# ====== 模型与路径配置 ======

# 基底模型（建议和你现在用的一致）
# 如果你用的是别的模型，这里改成对应 HF 名称即可
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 相对仓库根目录的路径（train_qlora.py 里会用 PROJECT_ROOT / DATA_PATH 拼起来）
# 也就是：<repo_root>/data/processed/merged_instructions.jsonl
DATA_PATH = "data/processed/merged_instructions.jsonl"

# 输出目录（同样是相对仓库根目录）
# 也就是：<repo_root>/outputs/qlora-qwen2.5-7b-sft
OUTPUT_DIR = "outputs/qlora-qwen2.5-7b-sft"


# ====== 数据量控制：sanity run / full training ======

# 如果只想做 sanity run（小样本快速跑通）：设成一个整数，比如 512
# 如果想用全部数据训练：改成 None
MAX_TRAIN_SAMPLES = None      # or None


# ====== 训练超参数（会传给 SFTConfig） ======

# 训练轮数
NUM_EPOCHS = 2              # sanity 跑建议 1；full 训练可以 2~3 自己调

# 单卡 batch size（显存不大就 1，A100 可以尝试 2~4）
PER_DEVICE_BATCH_SIZE = 4

# 梯度累积步数（显存不够就把这个设大一点）
GRAD_ACCUM_STEPS = 4

# 学习率（QLoRA 常见 2e-4 ~ 5e-5）
LEARNING_RATE = 1e-4

# 日志打印频率
LOGGING_STEPS = 10

# 保存策略："epoch" 表示每个 epoch 末尾保存一次
# 也可以用 "steps" 然后再配合 save_steps 自己在 train_qlora.py 里加
SAVE_STRATEGY = "epoch"

# 训练时最大序列长度（和模型上下文长度、显存有关，2048 比较稳）
MAX_SEQ_LENGTH = 2048

