## 連線到實驗室 Server

```
ssh -L 9090:localhost:8080 liyichen125@192.168.1.102
```

## 啟動虛擬環境

```
# 創建虛擬環境
python -m venv my_venv
# 啟用虛擬環境
source my_venv/bin/activate
```

## 安裝所需套件

```
pip install --upgrade pip
pip install -r requirements.txt
```

## Prepare Dataset

```
python prepare_ms_marco_v21.py
```
