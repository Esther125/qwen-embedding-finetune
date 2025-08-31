#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========= CONFIG =========
OUT_DIR = "./output-Qwen3-Embedding-0.6B"
LOSS_CSV_MAIN  = os.path.join(OUT_DIR, "loss_trend.csv")
LOSS_CSV_CLEAN = os.path.join(OUT_DIR, "loss_trend_clean.csv")
GPU_CSV        = os.path.join(OUT_DIR, "gpu_mem.csv")
LOSS_PNG       = os.path.join(OUT_DIR, "loss_trend.png")
GPU_PNG        = os.path.join(OUT_DIR, "gpu_mem_trend.png")

GPU_INDEX = 3   # 只畫第 3 張卡
# =========================

os.makedirs(OUT_DIR, exist_ok=True)

def load_loss_df():
    path = LOSS_CSV_CLEAN if os.path.exists(LOSS_CSV_CLEAN) else LOSS_CSV_MAIN
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 loss 檔案：{path}")
    df = pd.read_csv(path)
    for col in ["step","epoch","loss","eval_loss","eval_margin",
                "eval_mean_pos","eval_mean_neg","learning_rate",
                "memory_GiB","train_speed_iter_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    train_df = df[~df["loss"].isna()].sort_index().drop_duplicates(subset=["step"], keep="last")
    eval_df  = df[~df["eval_loss"].isna()].sort_index().drop_duplicates(subset=["step"], keep="last")
    merged = pd.merge(train_df[["step","loss"]],
                      eval_df[["step","eval_loss"]],
                      on="step", how="outer").sort_values("step").reset_index(drop=True)
    return merged, path

def plot_loss(df, out_path):
    if df.empty:
        print("[WARN] loss df 為空，跳過繪圖"); return
    plt.figure(figsize=(8,5))
    if df["loss"].notna().any():
        plt.plot(df["step"], df["loss"], label="train loss", alpha=0.35)
        # 移動平均更平滑（視需要可關掉）
        ma = df["loss"].rolling(window=21, min_periods=1, center=True).mean()
        plt.plot(df["step"], ma, label="train loss (MA-21)")
    if df["eval_loss"].notna().any():
        plt.scatter(df["step"], df["eval_loss"], label="eval loss")
    plt.title("Loss Trend")
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"[OK] saved: {out_path}")

def parse_mib(x):
    if isinstance(x, (int,float)): return float(x)
    if isinstance(x, str):
        x = x.strip().replace("MiB","").strip()
        try: return float(x)
        except: return np.nan
    return np.nan

def load_gpu_df():
    if not os.path.exists(GPU_CSV):
        raise FileNotFoundError(f"找不到 GPU 檔案：{GPU_CSV}")
    df = pd.read_csv(GPU_CSV)

    # 只保留指定 GPU index
    if "index" in df.columns:
        df["index"] = pd.to_numeric(df["index"], errors="coerce")
        df = df[df["index"] == GPU_INDEX].copy()
    else:
        # 若沒有 index 欄，直接全用（但通常有）
        print("[WARN] 找不到 'index' 欄，無法過濾 GPU。")

    # 解析時間與使用量
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.Series(range(len(df)))

    used_col = "memory.used" if "memory.used" in df.columns else \
               [c for c in df.columns if "memory" in c and "used" in c][0]
    df["memory.used.MiB"] = df[used_col].apply(parse_mib)
    df = df.dropna(subset=["memory.used.MiB"]).reset_index(drop=True)
    return df

def plot_gpu_mem(df, out_path):
    if df.empty:
        print("[WARN] gpu df 為空，跳過繪圖"); return
    idx = int(df["memory.used.MiB"].idxmax())
    peak_val = df.loc[idx, "memory.used.MiB"]
    peak_ts  = df.loc[idx, "timestamp"]

    plt.figure(figsize=(9,4.8))
    plt.plot(df["timestamp"], df["memory.used.MiB"], label="GPU memory used (MiB)")
    plt.scatter([peak_ts], [peak_val])
    plt.annotate(f"peak: {peak_val:.0f} MiB",
                 xy=(peak_ts, peak_val), xytext=(0,10), textcoords="offset points")
    plt.title(f"GPU Memory Usage (GPU {GPU_INDEX})")
    plt.xlabel("Time"); plt.ylabel("MiB")
    plt.grid(True); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"[OK] saved: {out_path}")
    print(f"[INFO] GPU{GPU_INDEX} peak usage: {peak_val:.0f} MiB at {peak_ts}")

def main():
    try:
        loss_df, used_path = load_loss_df()
        print(f"[INFO] loaded loss csv: {used_path} (rows={len(loss_df)})")
        plot_loss(loss_df, LOSS_PNG)
    except Exception as e:
        print("[ERR] loss plotting failed:", e)

    try:
        gpu_df = load_gpu_df()
        print(f"[INFO] loaded gpu csv: {GPU_CSV} (rows={len(gpu_df)})")
        plot_gpu_mem(gpu_df, GPU_PNG)
    except Exception as e:
        print("[ERR] gpu plotting failed:", e)

if __name__ == "__main__":
    main()
