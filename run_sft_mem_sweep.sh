#!/usr/bin/env bash
set -e

# 可調整
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
OUT_BASE=${OUT_BASE:-output}
MODEL=${MODEL:-"Qwen/Qwen3-Embedding-0.6B"}
LENS=("512" "1024" "2048")

# swift CLI fallback
if command -v swift >/dev/null 2>&1; then
  CMD=(swift sft)
else
  CMD=(python -m swift.cli.sft)
fi

STAMP=$(date +"%Y%m%d-%H%M%S")
OUT_DIR="${OUT_BASE}/sweep-${STAMP}"
mkdir -p "${OUT_DIR}"
SUMMARY_CSV="${OUT_DIR}/summary_peak_mem.csv"
echo "len,gpu_index,peak_used_mib,total_mib,util_percent,run_dir" > "${SUMMARY_CSV}"

for LEN in "${LENS[@]}"; do
  RUN_DIR="${OUT_DIR}/len-${LEN}"
  mkdir -p "${RUN_DIR}"
  LOG_CSV="${RUN_DIR}/gpu_mem.csv"

  # 啟動顯存採樣（每 500ms）
  nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total \
    --format=csv -lms 500 > "${LOG_CSV}" 2>&1 &
  NSMI=$!
  trap 'kill '"$NSMI"' 2>/dev/null || true' EXIT

  # 訓練（你的原始配置）
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  "${CMD[@]}" \
    --model "${MODEL}" \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --bf16 true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length "${LEN}" \
    --output_dir "${RUN_DIR}" \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name "swift-robot-len${LEN}" |& tee "${RUN_DIR}/train.log"

  # 關掉顯存採樣
  kill "$NSMI" 2>/dev/null || true

  # 計算峰值顯存並輸出
  PEAK_TXT="${RUN_DIR}/peak_mem.txt"
  awk -F, 'NR>1{
    gsub(/ MiB| MB/,"",$4); gsub(/ MiB| MB/,"",$5);
    idx=$2; used=$4+0; tot=$5+0;
    if (used>max[idx]) {max[idx]=used; total[idx]=tot}
  } END{
    for(i in max){
      util= (total[i]>0)? (100*max[i]/total[i]) : 0;
      printf("GPU %s peak_used=%d MiB / total=%d MiB (%.1f%%)\n", i, max[i], total[i], util);
    }
  }' "${LOG_CSV}" | tee "${PEAK_TXT}"

  # 追加到總表
  awk -F, -v L="${LEN}" -v RD="${RUN_DIR}" 'NR>1{
    gsub(/ MiB| MB/,"",$4); gsub(/ MiB| MB/,"",$5);
    if ($4+0 > p[$2]) { p[$2]=$4+0; t[$2]=$5+0 }
  } END{
    for(i in p){
      util=(t[i]>0)?(100*p[i]/t[i]):0;
      printf("%s,%s,%d,%d,%.2f,%s\n", L, i, p[i], t[i], util, RD);
    }
  }' "${LOG_CSV}" >> "${SUMMARY_CSV}"

done

echo
echo "✅ 完成！彙整表：${SUMMARY_CSV}"
cat "${SUMMARY_CSV}"
