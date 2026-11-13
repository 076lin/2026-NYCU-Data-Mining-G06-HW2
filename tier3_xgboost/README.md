# Tier 3：XGBoost（監督式）與 ECOD（非監督）

## 檔案

| 檔案 | 用途 |
|---|---|
| `prepare_data.py` | 產生 `artifacts/`（第一次執行，約 20 秒、1.3 GB） |
| `xgb_tier3.py` | XGBoost：訓練、在 val 決定 τ；`--test` 跑 test |
| `ecod_tier3.py` | ECOD：同上 |
| `common.py` | 共用的 τ 規則（val best-F1）、指標、計時 |
| `preprocess.py` | 與 `eda_and_preprocess/preprocess.py` 逐位元組相同，不要修改 |

## 環境與資料

1. 資料：把 CIC-IDS2017 的 `TrafficLabelling` 資料夾（8 個 CSV）放在這些程式**同一層**，
   或建 symlink：`ln -s <你的路徑>/TrafficLabelling TrafficLabelling`
2. 套件（Python 3.14）：
   - 在 repo 裡：根目錄 `uv sync`
   - 單獨資料夾：`uv venv --python 3.14 && uv pip install -r requirements.txt`
3. macOS 跑 XGBoost 需要 OpenMP：`brew install libomp`

## 執行（在這個資料夾）

```
uv run python prepare_data.py        # 一次
uv run python xgb_tier3.py           # 選參數 + 3 seeds，只跑 val
uv run python ecod_tier3.py          # 只跑 val
uv run python xgb_tier3.py --test    # test，只跑一次
uv run python ecod_tier3.py --test   # test，只跑一次
```

`--test` 會讀存好的模型和 val 的 τ，不會重新訓練。結果在 `results/xgb/`、`results/ecod/`。

## 設計重點

- **848,433**：`preprocess.py` 寫死 train benign 去重後 = 848,406，但 pandas 3.1.0
  實際是 848,433。`prepare_data.py` 在執行時覆寫這個值，`preprocess.py` 不改，其他 assert 照跑。
- **XGBoost**：`train_dedup` 依 Label 分層切 80% 訓練 / 20% holdout；holdout 用於
  early stopping（weighted log loss）與 grid（depth {4,6,8} × lr {0.05,0.1}），
  做法同 Xu & Liu (2025)。benign 與攻擊各占一半權重，5 種攻擊再平分。val 只用來決定 τ。
- **ECOD**：只用 `train_benign`；自己實作，ECDF 與偏態只在 train 上建立
  （PyOD 版本會把 val/test 併進去重算）。沒有超參數、結果是 deterministic。

## 訓練筆數（報告 §6）

| 模型 | 資料 | Normal | Anomaly |
|---|---|---|---|
| XGBoost 訓練 | `train_dedup` 80% | 678,746 | 153,345 |
| XGBoost holdout | `train_dedup` 20% | 169,687 | 38,336 |
| ECOD | `train_benign` | 848,433 | 0 |

`results/xgb_val_early_stop/` 是舊版（用 val 做 early stopping）：只長 2–4 棵樹，
因為越擬合已知攻擊，對 val 的未知攻擊越差。
