#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen3-Embedding-0.6B"
OUT_DIR="./output-$(basename "$MODEL")"
TRAIN_JSONL="/homepool2/liyichen125/workspace/qwen-embedding-finetune/data_prepared/train_infonce.jsonl"
VAL_JSONL="/homepool2/liyichen125/workspace/qwen-embedding-finetune/data_prepared/val_infonce.jsonl"

mkdir -p "$OUT_DIR"

# ===== GPU 記憶體紀錄 =====
GPU_CSV="${OUT_DIR}/gpu_mem.csv"
echo "timestamp,index,name,memory.used,memory.total" > "$GPU_CSV"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total \
    --format=csv,noheader -l 1 >> "$GPU_CSV" 2>&1 &
  NSMI_PID=$!
  trap 'kill ${NSMI_PID:-0} >/dev/null 2>&1 || true' EXIT
else
  echo "N/A,N/A,N/A,N/A,N/A" >> "$GPU_CSV"
  echo "[WARN] nvidia-smi 不存在，gpu_mem.csv 只會有佔位列"
fi

# ===== 訓練 =====
TRAIN_LOG="${OUT_DIR}/train_raw.log"
export PYTHONUNBUFFERED=1
swift sft \
  --model "$MODEL" \
  --task_type embedding \
  --model_type qwen3_emb \
  --train_type full \
  --dataset "$TRAIN_JSONL" \
  --val_dataset "$VAL_JSONL" \
  --columns '{"query":"query","response":"response"}' \
  --loss_type infonce \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 6e-6 \
  --eval_strategy steps \
  --eval_steps 200 \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 2 \
  --overwrite_output_dir \
  --logging_steps 50 \
  --report_to none \
  --lazy_tokenize false \
  --output_dir "$OUT_DIR" 2>&1 | tee -a "$TRAIN_LOG"

# ===== 訓練後：產生 loss_trend.csv（一定會有檔）=====
LOSS_CSV="${OUT_DIR}/loss_trend.csv"
OUT_DIR="$OUT_DIR" python - <<'PY'
import json, os, csv, glob, re
out_dir = os.environ.get("OUT_DIR", ".")
ts_path = os.path.join(out_dir, "trainer_state.json")
if not os.path.exists(ts_path):
    cks = sorted(glob.glob(os.path.join(out_dir, "checkpoint-*")), key=lambda p: int(p.split("-")[-1]))
    if cks:
        cand = os.path.join(cks[-1], "trainer_state.json")
        if os.path.exists(cand):
            ts_path = cand
rows = []
if os.path.exists(ts_path):
    with open(ts_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    for h in js.get("log_history", []):
        rows.append({
            "step": h.get("step") or h.get("global_step") or "",
            "loss": h.get("loss", ""),
            "eval_loss": h.get("eval_loss", "")
        })
else:
    # 後備：從 train_raw.log 解析
    log_path = os.path.join(out_dir, "train_raw.log")
    pat_step = re.compile(r'(?:global_)?step[= :]+(\d+)')
    pat_loss = re.compile(r'(^|[ ,])loss[= :]+([0-9]*\.[0-9]+|[0-9]+)')
    pat_eloss= re.compile(r'eval_loss[= :]+([0-9]*\.[0-9]+|[0-9]+)')
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s=l=el=""
                m = pat_step.search(line);  s = m.group(1) if m else ""
                m = pat_loss.search(line);  l = m.group(2) if m else ""
                m = pat_eloss.search(line); el= m.group(1) if m else ""
                if l or el:
                    rows.append({"step": s, "loss": l, "eval_loss": el})

csv_path = os.path.join(out_dir, "loss_trend.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as fw:
    w = csv.DictWriter(fw, fieldnames=["step","loss","eval_loss"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"[OK] wrote loss CSV -> {csv_path} ({len(rows)} rows)")
PY

echo "GPU memory usage CSV : $GPU_CSV"
echo "Training raw log     : $TRAIN_LOG"
echo "Loss trend CSV       : $LOSS_CSV"
