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
pip install -U datasets

```

## Prepare Dataset

```bash
python data_preprocess.py
```

新的資料集格式：

```json
{
  "query": "Does long-term oral melatonin administration reduce ethanol-induced increases in duodenal mucosal permeability and motility in rats?",
  "response": "Although further studies are needed, our data demonstrate that melatonin administration markedly improves duodenal barrier functions, suggesting its utility in clinical applications when intestinal barrier functions are compromised."
}
```

資料會存放在 /data_prepared 資料夾下

## Finetune

```bash
chmod +x run_finetune.sh

./run_finetune.sh
```