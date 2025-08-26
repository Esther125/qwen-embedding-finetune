import os, glob, json, random
import pandas as pd

DATA_DIR = "/homepool2/liyichen125/datasets/ms_marco/v2.1"   
OUT_DIR  = "./data_prepared"                                

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(42)

def extract_pairs_from_df(df, out_path):
    n = 0
    with open(out_path, "w", encoding="utf-8") as fw:
        for _, row in df.iterrows():
            query = str(row.get("query", "") or "").strip()
            passages = row.get("passages", None)

            if not query or passages is None:
                continue

            pos_texts, neg_texts = [], []

            cand = []
            if isinstance(passages, list):
                cand = passages
            elif isinstance(passages, dict):
                keys = passages.keys()
                L = len(passages.get("passage_text", []))
                for i in range(L):
                    cand.append({
                        "is_selected": passages.get("is_selected", [0]*L)[i],
                        "passage_text": passages.get("passage_text", [""]*L)[i],
                        "url": passages.get("url", [""]*L)[i] if "url" in keys else ""
                    })

            for p in cand:
                txt = (p.get("passage_text") or "").strip()
                if not txt:
                    continue
                if int(p.get("is_selected", 0)) == 1:
                    pos_texts.append(txt)
                else:
                    neg_texts.append(txt)

            # 正例：依序輸出
            for t in pos_texts:
                fw.write(json.dumps({"text1": query, "text2": t, "score": 1.0}, ensure_ascii=False) + "\n")
                n += 1

            # 沒正例就補一個負例，避免整列被丟掉
            if not pos_texts and neg_texts:
                t = random.choice(neg_texts)
                fw.write(json.dumps({"text1": query, "text2": t, "score": 0.0}, ensure_ascii=False) + "\n")
                n += 1
    print(f"[OK] wrote {n} pairs -> {out_path}")

def convert(split_glob, out_name):
    files = sorted(glob.glob(os.path.join(DATA_DIR, split_glob)))
    if not files:
        print(f"[WARN] no files for pattern: {split_glob}")
        return
    out_path = os.path.join(OUT_DIR, out_name)
    cnt = 0
    with open(out_path, "w", encoding="utf-8") as fw:
        pass
    for p in files:
        df = pd.read_parquet(p)
        extract_pairs_from_df(df, out_path)

convert("train-*.parquet",       "train.jsonl")
convert("validation-*.parquet",  "val.jsonl")
convert("test-*.parquet",        "test.jsonl")
