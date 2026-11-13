# Tier 2 kNN｜CIC-IDS2017
- results_val.xlsx：驗證集，門檻以各 seed 的最高 F1 決定。
- results_test.xlsx：週五測試，逐 seed 沿用已固定門檻。
- summary.json：逐 seed 的完整指標、各類攻擊計數與資源用量。
- locked_config.json：固定設定、門檻、資料切分指紋。
- validation_config.json：重現程式所需的驗證結果與門檻。
- *.py：重現程式與依賴模組。

三個 seeds 共用資料切分，每次只從去重後的正常訓練流量抽取 32,768 筆，不放回抽樣。異常分數是到第 5 個最近鄰居的歐氏距離。AP 使用 average_precision_score；表格標準差是三次的樣本標準差。兩項 kNN 消融尚未執行，本次使用原先驗證過的設定。

重現時把 Python 檔和 validation_config.json 放在共用 eda_and_preprocess 目錄（內有 artifacts/X_scaled.npy、artifacts/meta.parquet、artifacts/splits.npz）。Linux 環境需有 PyTorch CUDA、NumPy、pandas、pyarrow、scikit-learn、joblib。已完成的正式測試留存在 NAS 的 tier2_runs/knn_final，該目錄中的 locked_config.json 避免同目錄誤重跑。ZIP 不包含原始流量資料。

重現指令（範例的 GPU 2 須有至少 6 GiB 可用顯存；只在需要重現時使用）：

```sh
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  taskset -c 0-3 python tier2_knn_final.py --validation validation_config.json --output tier2_runs/knn_reproduction
```

來源：UNB CIC-IDS2017 https://www.unb.ca/cic/datasets/ids-2017.html
