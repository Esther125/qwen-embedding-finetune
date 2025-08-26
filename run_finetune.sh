set -e

MODEL=${1:-"Qwen/Qwen3-Embedding-0.6B"}   # 預設 0.6B
SEQ_LEN=${2:-1024}                        # 預設 1024
OUT_DIR="./output-$(basename $MODEL)-len${SEQ_LEN}"

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
  --bf16
