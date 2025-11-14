BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

DATA_PATH = "data/processed/merged_instructions.jsonl"
OUTPUT_DIR = "outputs/qlora-qwen2.5-7b-sft"

# 先做一个小规模 sanity run，可以后面再调大
MAX_TRAIN_SAMPLES = 2000  # None 代表用全部数据

# 序列长度 & batch
MAX_SEQ_LENGTH = 1024
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8

NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
LOGGING_STEPS = 50
SAVE_STRATEGY = "epoch"
