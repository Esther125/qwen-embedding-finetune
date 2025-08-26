set -e

MODEL=${1:-"Qwen/Qwen3-Embedding-0.6B"}
SEQ_LEN=${2:-1024}
OUT_DIR="./output-$(basename $MODEL)-len${SEQ_LEN}"

# 開始紀錄 GPU 使用情況，每秒寫一次到檔案
LOGFILE="${OUT_DIR}/gpu_mem.log"
mkdir -p $OUT_DIR
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total --format=csv -l 1 > $LOGFILE 2>&1 &
NSMI_PID=$!

# === 執行訓練 ===
swift sft \
  --model "$MODEL" \
  --task_type embedding \
  --model_type qwen3_emb \
  --train_type full \
  --dataset "jsonl:./data_prepared/train.jsonl:pair" \
  --val_dataset "jsonl:./data_prepared/val.jsonl:pair" \
  --split_dataset_ratio 0.05 \
  --eval_strategy steps \
  --output_dir "$OUT_DIR" \
  --eval_steps 200 \
  --save_steps 200 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 6e-6 \
  --loss_type cosine_similarity \
  --drop_last true \
  --max_seq_len $SEQ_LEN \
  --bf16 true

# 訓練結束後停止 nvidia-smi 紀錄
kill $NSMI_PID

echo "GPU memory log saved at $LOGFILE"
