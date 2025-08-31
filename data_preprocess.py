
# ===================== CONFIG =====================
# 資料來源
DATASET   = "qiaojin/PubMedQA"
CONFIG    = "pqa_artificial"   
SPLIT     = "train"
STREAMING = False             

# 數量 / 顯示
TAKE_N    = 20_000            # 只拿前 N 筆
PRINT_N   = 5                 # 印出前幾筆（raw & processed）供確認

# 輸出與切分
OUT_DIR      = "./data_prepared"
TRAIN_RATIO  = 0.90
VAL_RATIO    = 0.05
TEST_RATIO   = 0.05
SEED         = 42

# 清理規則
MIN_QUERY_LEN    = 1           # 最低 query 字元數（避免空 query）
# ==================================================

import os
import re
import json
import random
from typing import List, Dict
from datasets import load_dataset

# ---------- 基本清理：轉字串、strip、壓縮空白 ----------
_WS_RE = re.compile(r"\s+")

def _as_clean_text(x) -> str:
    """轉成字串、去頭尾空白、把多重空白/換行壓成單一空白。"""
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    x = x.strip()
    x = _WS_RE.sub(" ", x)
    return x

# -------------- core methods --------------
def download_dataset() -> List[Dict]:
    """
    從 HF 下載資料，只取前 TAKE_N 筆。
    - STREAMING=True：串流收集前 N
    - STREAMING=False：下載完整 split，再擷取前 N
    回傳：list of raw rows (dict)
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    if CONFIG:
        ds = load_dataset(DATASET, CONFIG, split=SPLIT, streaming=STREAMING)
    else:
        ds = load_dataset(DATASET, split=SPLIT, streaming=STREAMING)

    rows: List[Dict] = []
    if STREAMING:
        for i, row in enumerate(ds):
            rows.append(row)
            if i + 1 >= TAKE_N:
                break
    else:
        n = min(TAKE_N, len(ds))
        for i in range(n):
            rows.append(ds[i])

    print(f"[INFO] Downloaded {len(rows)} rows from {DATASET}/{CONFIG or '(no-config)'}/{SPLIT}")
    for i, r in enumerate(rows[:PRINT_N]):
        print(f"\n--- RAW row {i} ---")
        print(json.dumps(r, ensure_ascii=False, indent=2))
    return rows

def data_preprocess(raw_rows: List[Dict]) -> List[Dict]:
    """
    嚴格轉成 InfoNCE 格式：
      {"query": <question>, "response": <long_answer>}
    - 僅保留這兩個 key
    - 兩欄皆進行基本清理
    """
    examples: List[Dict] = []
    skipped = 0

    for row in raw_rows:
        q  = _as_clean_text(row.get("question", ""))
        la = _as_clean_text(row.get("long_answer", ""))

        if len(q) < MIN_QUERY_LEN:
            skipped += 1
            continue

        examples.append({"query": q, "response": la})

    print(f"[INFO] Preprocessed -> kept {len(examples)} / skipped {skipped}")
    for i, ex in enumerate(examples[:PRINT_N]):
        print(f"\n--- PROC row {i} ---")
        print(json.dumps(ex, ensure_ascii=False, indent=2))
    return examples

def output_dataset(examples: List[Dict]) -> None:
    """
    打亂並切分 examples，輸出 train/val/test 到 OUT_DIR 下的 .jsonl
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED)
    rng.shuffle(examples)

    n = len(examples)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_test  = n - n_train - n_val

    train_set = examples[:n_train]
    val_set   = examples[n_train:n_train + n_val]
    test_set  = examples[n_train + n_val:]

    def _write_jsonl(path: str, items: List[Dict]):
        with open(path, "w", encoding="utf-8") as fw:
            for ex in items:
                fw.write(json.dumps({"query": ex["query"], "response": ex["response"]}, ensure_ascii=False) + "\n")
        print(f"[OK] wrote {len(items):>6} -> {path}")

    _write_jsonl(os.path.join(OUT_DIR, "train_infonce.jsonl"), train_set)
    _write_jsonl(os.path.join(OUT_DIR, "val_infonce.jsonl"),   val_set)
    _write_jsonl(os.path.join(OUT_DIR, "test_infonce.jsonl"),  test_set)

# ----------------- main ------------------
def main():
    raw_rows = download_dataset()
    examples = data_preprocess(raw_rows)
    output_dataset(examples)

if __name__ == "__main__":
    main()
