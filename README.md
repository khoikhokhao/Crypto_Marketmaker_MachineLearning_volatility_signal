# CAKEUSDT 10s Volatility Classifier (Order Book Based)

This repo contains an end-to-end pipeline to detect short-term volatility bursts on **CAKEUSDT (Binance spot)** using **order book (LOB) data + tree models (Random Forest / LightGBM / XGBoost)**.

Mục tiêu: tại mỗi thời điểm, dự đoán xem **trong 10 giây tới giá có “nhảy mạnh” hay không**, để phục vụ cho chiến lược **market making / alpha scalping** (tối ưu quote, tránh bị “đập” khi vol tăng đột ngột, v.v.).

---

## 1. Problem setup

- Universe: `CAKEUSDT` spot, Binance.
- Raw input: level-5 order book snapshots:
  - `ts_ms, ts_iso, best_bid, best_ask, last_price, best_bid_qty, best_ask_qty, bid_notional_5, ask_notional_5`
- Horizon: **10 seconds** ahead.
- Target label (binary):
  - `1` = trong 10s tiếp theo, biên độ dao động giá ≥ `0.002` (theo mid hoặc last_price, consistent với notebook).
  - `0` = ngược lại (phiên “êm”, không biến động mạnh).
- Nhãn chỉ được gán cho các timestamp “đủ dữ liệu” và **không nằm trong gap lớn** (sử dụng `label_valid` + `max_future_gap10s` logic giống notebook offline).

---

## 2. Offline pipeline (notebook → model .pkl)

Dữ liệu lịch sử được xử lý bằng notebook (không nằm trong repo hoặc được rút gọn):

1. **Load & EDA**
   - Đọc file raw `CAKEUSDT_binance_lob_fn.csv` (~800k dòng).
   - Kiểm tra NaN, spread, outlier, gap thời gian, phân bố daily.

2. **Resample 1s + label**
   - Chuyển từ event-based → time-based 1s:
     - Floor theo giây, lấy **last tick mỗi giây**, sau đó reindex full timeline và **ffill** LOB.
   - Tính các trường:
     - `has_tick`, `gap_len`, `range_10s`, `max_future_gap10s`.
   - Gán nhãn `label_move10s_rng0p002` (0/1) cho các hàng `label_valid = 1`.

3. **Feature engineering**
   Trên chuỗi 1s đã được làm sạch, build các feature (giống pipeline offline trước đó):

   - Giá & biến động:
     - `mid = (best_bid + best_ask)/2`
     - `spread = best_ask - best_bid`
     - `ret_1s`, `ret_2s`, `ret_5s` (log-return)
     - `rv_10s` (realized volatility 10s)
     - `range_10s_cur` (hi-lo range trong 10s gần nhất)
   - Order book depth:
     - `depth_bid`, `depth_ask`, `depth_sum`, `depth_imb`
     - `qty_bid`, `qty_ask`, `qty_sum`, `qty_imb`
   - Time-of-day:
     - `tod_sin`, `tod_cos` (encoding chu kỳ 24h)
   - Các cột meta: `has_tick`, `gap_len`.

   File tổng hợp:  
   `D:\MM\data\CAKEUSDT_binance_lob_fn_features_insample_oos.csv`

4. **Train/Val/Test/OOS split**
   - In-sample date range: **24/11 – 01/12** (UTC), chia thành:
     - Train / Val / Test theo time-series split.
   - **OOS**: full ngày **02/12** (UTC).
   - Duy trì **class imbalance thật** (~4–7% label=1 tùy ngày).

5. **Modeling**
   - Thử nhiều model: LightGBM, XGBoost, RandomForest, ExtraTrees, Logistic Regression, MLP…
   - Đánh giá bằng ROC-AUC, PR-AUC, F1 trên VAL, TEST, OOS.
   - Chọn 3 tree models chính:
     - **LightGBM**
     - **XGBoost**
     - **RandomForest**
   - Scan threshold trên VAL (0.1 → 0.9) theo F1, sau đó **re-check trên TEST + OOS**.

6. **Final choice for production**
   - Vì mục tiêu là **“bắt được vol”** (ưu tiên recall, chấp nhận nhiều tín hiệu hơn):
     - Chọn:
       - LightGBM @ threshold ≈ 0.80 (version baseline vol-10s).
       - RandomForest @ threshold ≈ 0.85.
   - Hai model được lưu lại để dùng sau trong MM engine:

     ```text
     D:\MM\models_vol\lgbm_vol_10s.pkl
     D:\MM\models_vol\rf_vol_10s.pkl
     ```

---

## 3. Online inference – RF vol 10s (Binance realtime)

File chính:  
`binance_vol10s_rf_func.py` (và/hoặc `binance_vol10s_rf_stream.py` tùy bản bạn public).

Có 2 chế độ:

### 3.1. Function mode (dùng trong MM engine)

Core API:

```python
def is_volatility(
    lob_data: pd.DataFrame,
    model_path: str = r"D:\MM\models_vol\rf_vol_10s.pkl",
    threshold: float = 0.85,
    verbose: bool = False,
) -> bool:
    """
    lob_data: DataFrame với cột:
        ts_ms, ts_iso,
        best_bid, best_ask, last_price,
        best_bid_qty, best_ask_qty,
        bid_notional_5, ask_notional_5

    - Tự sort theo thời gian
    - Tự resample 1s + ffill
    - Tự tính full bộ feature giống offline
    - Lấy hàng cuối cùng, predict p(move_10s) bằng RF
    - So sánh với threshold, trả về True/False:
        True  = dự đoán có “vol move” trong 10s tới
        False = dự đoán không
    """
```

## 4. 👨‍💻 Tác giả
**Phạm Minh Khôi**  
FPT University – AI Major  
📧 Contact: [phamminhkhoi.05.09.12@gmail.com]  
📦 HuggingFace: * (https://huggingface.co/minhkhoiaifinance/Crypto-Marketmaker-MachineLearning-volatility-signal) *
