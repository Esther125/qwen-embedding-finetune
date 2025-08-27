## 連線到實驗室 Server

```bash
ssh -L 9090:localhost:8080 liyichen125@192.168.1.102
```

## 啟動虛擬環境

```bash
# 創建虛擬環境
python -m venv my_venv
# 啟用虛擬環境
source my_venv/bin/activate
```

## 安裝所需套件

```bash
pip install --upgrade pip
pip install pandas pyarrow
pip install torch transformers accelerate datasets evaluate scikit-learn huggingface_hub ms-swift

```

## Prepare Dataset

```bash
python prepare_ms_marco_v21.py
```

將 MS MARCO v2.1 的資料結構轉成 pair + label

-   label = 1.0 (正例)
    當 is_selected == 1 → 表示這段 passage 被標記為正確回答了 query。
-   label = 0.0 (負例)
    當 is_selected == 0 → 表示 passage 跟 query 不相關。

新的資料集格式：

```json
{
    "text1": "differentiate between population density and population distribution.",
    "text2": "Population distribution is the way in which people are spread across a given area, whereas population density is the average number of people per square kilometre. It's basically a way of measuring the population distribution. Hope this helps.",
    "label": 1.0
}
```

資料會存放在 /data_prepared 資料夾下

## Finetune

```bash
chmod +x run_finetune.sh

# 小模型 + 512 context
./run_finetune.sh Qwen/Qwen3-Embedding-0.6B 512

# 小模型 + 1024 context
./run_finetune.sh Qwen/Qwen3-Embedding-0.6B 1024

# 小模型 + 2048 context
./run_finetune.sh Qwen/Qwen3-Embedding-0.6B 2048

```
