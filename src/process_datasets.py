import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def process_alpaca():
    in_path = os.path.join(RAW_DIR, "alpaca.jsonl")
    out_path = os.path.join(PROC_DIR, "alpaca_proc.jsonl")
    print(f"Processing Alpaca: {in_path} -> {out_path}")

    with open(out_path, "w", encoding="utf-8") as fout:
        for ex in load_jsonl(in_path):
            inst = ex.get("instruction", "")
            inp = ex.get("input", "") or ""
            out = ex.get("output", "")

            rec = {
                "instruction": inst,
                "input": inp,
                "output": out,
                "source": "alpaca",
                "category": "general",
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")


def process_dolly():
    in_path = os.path.join(RAW_DIR, "dolly.jsonl")
    out_path = os.path.join(PROC_DIR, "dolly_proc.jsonl")
    print(f"Processing Dolly: {in_path} -> {out_path}")

    with open(out_path, "w", encoding="utf-8") as fout:
        for ex in load_jsonl(in_path):
            inst = ex.get("instruction", "")
            ctx = ex.get("context", "") or ""
            resp = ex.get("response", "")
            cat = ex.get("category", "general")

            rec = {
                "instruction": inst,
                "input": ctx,
                "output": resp,
                "source": "dolly",
                "category": cat or "general",
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")


def process_gsm8k():
    in_path = os.path.join(RAW_DIR, "gsm8k.jsonl")
    out_path = os.path.join(PROC_DIR, "gsm8k_proc.jsonl")
    print(f"Processing GSM8K: {in_path} -> {out_path}")

    with open(out_path, "w", encoding="utf-8") as fout:
        for ex in load_jsonl(in_path):
            q = ex.get("question", "")
            a = ex.get("answer", "")

            rec = {
                "instruction": q,
                "input": "",
                "output": a,
                "source": "gsm8k",
                "category": "math",
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")


def process_self_instruct():
    in_path = os.path.join(RAW_DIR, "self_instruct.jsonl")
    out_path = os.path.join(PROC_DIR, "self_instruct_proc.jsonl")
    print(f"Processing Self-Instruct: {in_path} -> {out_path}")

    with open(out_path, "w", encoding="utf-8") as fout:
        for ex in load_jsonl(in_path):
            # 自指令数据里通常有 prompt + instances 列表
            prompt = ex.get("prompt", "")
            instances = ex.get("instances", [])
            src = ex.get("source", "self_instruct")

            # 每个 instance 单独展开成一条样本
            for inst_obj in instances:
                inst = inst_obj.get("instruction", prompt)
                inp = inst_obj.get("input", "") or ""
                out = inst_obj.get("output", "")

                rec = {
                    "instruction": inst,
                    "input": inp,
                    "output": out,
                    "source": src,
                    "category": "self_instruct",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")


def merge_all():
    # 合并四个处理后的文件到一个总的训练集
    out_path = os.path.join(PROC_DIR, "merged_instructions.jsonl")
    print(f"Merging all processed datasets into: {out_path}")

    files = [
        "alpaca_proc.jsonl",
        "dolly_proc.jsonl",
        "gsm8k_proc.jsonl",
        "self_instruct_proc.jsonl",
    ]

    count = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for fname in files:
            fpath = os.path.join(PROC_DIR, fname)
            if not os.path.exists(fpath):
                print(f"  [WARN] File not found, skip: {fpath}")
                continue
            for ex in load_jsonl(fpath):
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                count += 1

    print(f"✅ Merged total samples: {count}")


def main():
    os.makedirs(PROC_DIR, exist_ok=True)
    process_alpaca()
    process_dolly()
    process_gsm8k()
    process_self_instruct()
    merge_all()


if __name__ == "__main__":
    main()
