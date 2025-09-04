import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import random

# ---------- utils ----------
def load_dataset(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def _encode_norm(model, texts, batch_size=64, show_bar=False):
    # 統一做 L2 normalize，之後點積=cosine
    try:
        emb = model.encode(texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=show_bar)
    except TypeError:
        emb = model.encode(texts, batch_size=batch_size, show_progress_bar=show_bar)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return emb

# ---------- Hard negatives by semantic similarity (with base model) ----------
def _precompute_all_doc_emb(base_model, data):
    docs = [row["response"] for row in data]
    doc_emb = _encode_norm(base_model, docs, show_bar=True)
    return docs, doc_emb  # docs: list[str], doc_emb: (N, d)

def _pick_hard_neg_indices_for_query(base_model, doc_emb, data, i, k_neg=9):
    """
    用 base_model：把第 i 題的 query 跟全資料 response 算相似度，
    取與正解不同的前 k_neg 個 index。
    """
    q = data[i]["query"]
    q_emb = _encode_norm(base_model, [q])  # (1, d)
    sims = (q_emb @ doc_emb.T).ravel()     # (N,)
    order = np.argsort(-sims)              # 大→小
    # 跳過正解（i）
    neg_list = []
    for j in order:
        if j == i:
            continue
        neg_list.append(j)
        if len(neg_list) >= k_neg:
            break
    return neg_list  # 長度 k_neg

# ---------- Evaluation: Top-1 / Top-3 with hard negatives ----------
def evaluate_accuracy(model: SentenceTransformer, data: list, k_neg: int = 9, repeats: int = 1, seed: int = 42,
                      base_model_for_neg=None, cached_docs=None, cached_doc_emb=None):
    """
    語意最接近的 hard negatives 評估（每題 1 正解 + k_neg 硬負樣本；不再隨機）
    回傳 Top-1 與 Top-3 accuracy
    - base_model_for_neg: 用哪個模型來挑 hard negatives（建議用 base_model）
    - cached_docs, cached_doc_emb: 用 _precompute_all_doc_emb() 的結果避免重算
    """
    assert base_model_for_neg is not None, "請把 base_model 傳進來當 hard negatives 的挑選模型"
    assert cached_docs is not None and cached_doc_emb is not None, "請先預編碼所有 response"

    top1_correct = 0
    top3_correct = 0
    total = 0
    N = len(data)

    for i in tqdm(range(N), desc=f"eval[hard k_neg={k_neg}]"):
        # 固定同一組 candidates：gold + 語意最像的負樣本（用 base_model 選）
        neg_idxs = _pick_hard_neg_indices_for_query(base_model_for_neg, cached_doc_emb, data, i, k_neg=k_neg)
        cand_idxs = [i] + neg_idxs
        candidates = [cached_docs[j] for j in cand_idxs]

        # encode query / candidates with "model"（要被評估的模型）
        q_emb = _encode_norm(model, [data[i]["query"]])
        d_emb = _encode_norm(model, candidates)

        sims = (q_emb @ d_emb.T).ravel()  # (1+k_neg,)
        order = np.argsort(-sims)         # 大→小
        true_index = 0                    # gold 在 cand_idxs 的第 0 個

        if order[0] == true_index:
            top1_correct += 1
        if true_index in order[:3]:
            top3_correct += 1
        total += 1

    top1_acc = top1_correct / total if total > 0 else 0.0
    top3_acc = top3_correct / total if total > 0 else 0.0
    return top1_acc, top3_acc

# ---------- Examples viewer with SAME candidates for both models ----------
def show_examples(base_model, finetuned_model, data, num_examples=2, seed=42, k_neg=9,
                  cached_docs=None, cached_doc_emb=None):
    """
    抽 num_examples 筆樣本，對每題先用 base_model 固定 1 正解 + k_neg 個「語意最接近」的負樣本（同一組），
    然後用同一組 candidates 比較 Base 與 Finetuned 的 Top-3（輸出格式沿用你原本的風格）。
    """
    rng = random.Random(seed)
    N = len(data)
    chosen = rng.sample(range(N), min(num_examples, N))

    for idx in chosen:
        query = data[idx]["query"]
        gold  = data[idx]["response"]

        # 用 base_model 從全庫挑語意最像的負樣本（固定 candidates）
        neg_idxs = _pick_hard_neg_indices_for_query(base_model, cached_doc_emb, data, idx, k_neg=k_neg)
        cand_idxs = [idx] + neg_idxs
        candidates = [cached_docs[j] for j in cand_idxs]

        print("="*80)
        print(f"Query: {query}")
        print(f"Gold : {gold}\n")

        def rank_top3(model, name):
            q = _encode_norm(model, [query])
            d = _encode_norm(model, candidates)
            sims = (q @ d.T).ravel()
            order = np.argsort(-sims)[:3]
            print(f"[{name} Model Top-3]  (同一組 candidates)")
            for r, j in enumerate(order, 1):
                tag = " (TRUE)" if j == 0 else ""
                print(f"  #{r} sim={sims[j]:.4f}{tag}")
                print("     ", candidates[j][:200].replace("\n", " "))
            print()

        # 用同一組 candidates 分別排名
        rank_top3(base_model, "Base")
        rank_top3(finetuned_model, "Finetuned")

# ---------- main ----------
def main():
    data_path = "./data_prepared/test_infonce.jsonl"
    data = load_dataset(data_path)

    # Base model（也用來挑 hard negatives）
    base_model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"device_map": {"": 0}},
        tokenizer_kwargs={"padding_side": "left"},
    )

    # Finetuned model（你匯出的 ST 版本）
    finetuned_model = SentenceTransformer(
        "./exported-model",
        model_kwargs={"device_map": {"": 0}},
        tokenizer_kwargs={"padding_side": "left"},
    )

    # 預先把所有 response 向量化一次（用 base_model）
    docs, doc_emb = _precompute_all_doc_emb(base_model, data)

    # 先印 2 個例子，確認候選一致且排序差異
    show_examples(base_model, finetuned_model, data, num_examples=2, k_neg=9,
                  cached_docs=docs, cached_doc_emb=doc_emb)

    # 再做整體 hard negatives 評估
    print("\n[Base Model 評估中 - hard negatives]")
    base_top1, base_top3 = evaluate_accuracy(base_model, data, k_neg=9, repeats=1, seed=42,
                                             base_model_for_neg=base_model, cached_docs=docs, cached_doc_emb=doc_emb)

    print("\n[Finetuned Model 評估中 - hard negatives]")
    ft_top1, ft_top3 = evaluate_accuracy(finetuned_model, data, k_neg=9, repeats=1, seed=42,
                                         base_model_for_neg=base_model, cached_docs=docs, cached_doc_emb=doc_emb)

    print("\n========= 評估結果（hard negatives）=========")
    print(f"[Base Model]")
    print(f"Top-1 Accuracy: {base_top1:.4f}")
    print(f"Top-3 Accuracy: {base_top3:.4f}")

    print(f"\n[Finetuned Model]")
    print(f"Top-1 Accuracy: {ft_top1:.4f}")
    print(f"Top-3 Accuracy: {ft_top3:.4f}")
    print("============================================")

if __name__ == "__main__":
    main()
