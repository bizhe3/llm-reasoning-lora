import os
from datasets import load_dataset

RAW_DIR = "data/raw"

# 要下载的数据集：名字 -> (HF 路径, 子配置名)
DATASETS = {
    # "alpaca": ("tatsu-lab/alpaca", None),
    # "dolly": ("databricks/databricks-dolly-15k", None),
    # "gsm8k": ("gsm8k", "main"),             # GSM8K 需要指定 "main"
    # "Open-Orca": ("Open-Orca/OpenOrca", None),
    "self_instruct": ("yizhongw/self_instruct", "super_natural_instructions")
}

def show_samples(ds, name, n=3):
    print(f"\n====== {name.upper()}：前 {n} 条样本 ======")
    for i in range(min(n, len(ds))):
        print(ds[i])
        print("-----------------------------------")

def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    for alias, (path, subset) in DATASETS.items():
        print(f"\n📥 正在下载数据集：{alias} ({path}) ...")
        if subset is not None:
            dataset = load_dataset(path, subset)
        else:
            dataset = load_dataset(path)

        train_split = dataset["train"]

        # 保存为本地 jsonl
        save_path = os.path.join(RAW_DIR, f"{alias}.jsonl")
        print(f"💾 保存到：{save_path}")
        train_split.to_json(save_path)

        # 打印前几条看看结构
        show_samples(train_split, alias, n=2)

    print("\n✅ 全部数据集下载 & 保存完成！")

if __name__ == "__main__":
    main()
