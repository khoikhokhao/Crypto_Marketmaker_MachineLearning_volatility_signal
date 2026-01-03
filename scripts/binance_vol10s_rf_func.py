# D:\MM\binance_vol10s_rf_func.py
"""
Dự đoán xem trong 10s tới có khả năng xảy ra "vol move" hay không,
dựa trên RandomForest vol_10s đã train offline.

API chính:

    is_volatility(lob_data, threshold=0.85, verbose=True)

Trong đó:
- lob_data: pandas.DataFrame với các cột:
    ts_ms, ts_iso, best_bid, best_ask, last_price,
    best_bid_qty, best_ask_qty, bid_notional_5, ask_notional_5
  Mỗi dòng tương ứng 1 snapshot 1s gần nhất (hoặc gần giống 1s).
  len(lob_data) phải >= số giây lịch sử tối thiểu để build feature.

- threshold: ngưỡng p(move_10s) để classify (mặc định 0.85, giống RF best trên val).
- verbose: in ra trạng thái / debug nếu cần.

Trả về:
- True  => model dự đoán "có khả năng VOL MOVE trong 10s tới".
- False => model dự đoán "không move / hoặc chưa đủ điều kiện để predict".
"""
from __future__ import annotations
import os
from typing import Tuple, Optional, Union

import numpy as np
import pandas as pd
from joblib import load
from pandas.api.types import is_datetime64_any_dtype


# ====================== CONFIG & GLOBALS ======================

MODEL_PATH_RF = r"D:\MM\models_vol\rf_vol_10s.pkl"

# Các feature giống file feature offline (19 cột):
FEATURE_COLS = [
    "mid",
    "spread",
    "ret_1s",
    "ret_2s",
    "ret_5s",
    "rv_10s",
    "range_10s_cur",
    "depth_bid",
    "depth_ask",
    "depth_sum",
    "depth_imb",
    "qty_bid",
    "qty_ask",
    "qty_sum",
    "qty_imb",
    "has_tick",
    "gap_len",
    "tod_sin",
    "tod_cos",
]

# Số history tối thiểu để feature ổn (ret_5s + rv_10s + range_10s_cur)
MIN_HISTORY_ROWS = 30  # ~30s

# Nếu gap_len cuối cùng > 5s thì cảnh báo / không predict
MAX_ALLOWED_GAP_SEC = 5.0

# cache model để không load lại mỗi lần
_RF_MODEL = None


# ====================== HELPER: LOAD MODEL ======================

def _load_rf_model(model_path: str = MODEL_PATH_RF):
    """
    Load RF model vol_10s, cache vào global.
    """
    global _RF_MODEL
    if _RF_MODEL is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy model RF: {model_path}")
        print(f"[is_volatility] Loading RF vol model from: {model_path}")
        _RF_MODEL = load(model_path)
        print("[is_volatility] RF model loaded.")
    return _RF_MODEL


# ====================== HELPER: CHUẨN HÓA INPUT ======================

def _prepare_lob_df(lob_data: Union[pd.DataFrame, list, tuple]) -> pd.DataFrame:
    """
    Đảm bảo lob_data là DataFrame đúng cột, sort theo thời gian.
    """

    expected_cols = [
        "ts_ms",
        "ts_iso",
        "best_bid",
        "best_ask",
        "last_price",
        "best_bid_qty",
        "best_ask_qty",
        "bid_notional_5",
        "ask_notional_5",
    ]

    if isinstance(lob_data, pd.DataFrame):
        df = lob_data.copy()
    else:
        # giả sử là list/tuple of dicts / tuples theo đúng thứ tự
        df = pd.DataFrame(lob_data, columns=expected_cols)

    # check cột
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"lob_data thiếu cột: {missing}")

    # parse ts_iso (dùng pandas, không dùng np.issubdtype)
    if not is_datetime64_any_dtype(df["ts_iso"]):
        df["ts_iso"] = pd.to_datetime(df["ts_iso"], utc=True, errors="coerce")

    # sort theo thời gian
    df = df.sort_values("ts_iso").reset_index(drop=True)

    # đảm bảo ts_ms tồn tại (nếu thiếu thì derive từ ts_iso)
    if df["ts_ms"].isna().any():
        df["ts_ms"] = (df["ts_iso"].view("int64") // 10**6).astype("int64")

    return df


# ====================== HELPER: BUILD FEATURE ======================

def _build_feature_from_lob(df_lob: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Từ df_lob (đã chuẩn hóa), tính toàn bộ feature giống offline,
    trả về:
        - feat_df: DataFrame shape (1, n_features) cho dòng cuối cùng
        - status: "ok" hoặc lý do không predict (not_enough_history, gap_too_large, nan_in_features)
    """

    df = df_lob.copy()

    # ---- basic market features ----
    df["mid"] = (df["best_bid"] + df["best_ask"]) / 2.0
    df["spread"] = df["best_ask"] - df["best_bid"]

    # returns trên mid (giống kiểu đã làm offline: pct_change)
    df["ret_1s"] = df["mid"].pct_change()
    df["ret_2s"] = df["mid"].pct_change(2)
    df["ret_5s"] = df["mid"].pct_change(5)

    # realized vol 10s (std của ret_1s trong 10s)
    df["rv_10s"] = df["ret_1s"].rolling(window=10, min_periods=5).std()

    # range_10s_cur: (max-mid - min-mid)/mid trong cửa sổ 10s
    roll_max = df["mid"].rolling(window=10, min_periods=5).max()
    roll_min = df["mid"].rolling(window=10, min_periods=5).min()
    df["range_10s_cur"] = (roll_max - roll_min) / df["mid"].replace(0, np.nan)

    # depth & imbalance (dùng notional 5 levels)
    df["depth_bid"] = df["bid_notional_5"]
    df["depth_ask"] = df["ask_notional_5"]
    df["depth_sum"] = df["depth_bid"] + df["depth_ask"]
    denom_depth = df["depth_sum"].replace(0, np.nan)
    df["depth_imb"] = (df["depth_bid"] - df["depth_ask"]) / denom_depth

    # quantity side
    df["qty_bid"] = df["best_bid_qty"]
    df["qty_ask"] = df["best_ask_qty"]
    df["qty_sum"] = df["qty_bid"] + df["qty_ask"]
    denom_qty = df["qty_sum"].replace(0, np.nan)
    df["qty_imb"] = (df["qty_bid"] - df["qty_ask"]) / denom_qty

    # has_tick & gap_len:
    #  - Giả định lob_data là dòng "thật" (từ WS), không phải ffill => has_tick = 1
    #  - gap_len đo thời gian kể từ lần cuối có update nếu dt_sec > 2s
    df["has_tick"] = 1
    dt_sec = df["ts_iso"].diff().dt.total_seconds().fillna(1.0)

    gap = np.zeros(len(df))
    for i in range(1, len(df)):
        if dt_sec.iat[i] > 2.0:
            gap[i] = gap[i - 1] + dt_sec.iat[i]
        else:
            gap[i] = 0.0
    df["gap_len"] = gap

    # time-of-day features
    t = df["ts_iso"].dt
    seconds_in_day = 24 * 60 * 60
    seconds = t.hour * 3600 + t.minute * 60 + t.second
    angle = 2 * np.pi * seconds / seconds_in_day
    df["tod_sin"] = np.sin(angle)
    df["tod_cos"] = np.cos(angle)

    # ---- kiểm tra điều kiện history ----
    if len(df) < MIN_HISTORY_ROWS:
        return None, f"not_enough_history (need >= {MIN_HISTORY_ROWS}, got {len(df)})"

    # ---- kiểm tra gap cuối cùng ----
    last_gap = float(df["gap_len"].iloc[-1])
    if last_gap > MAX_ALLOWED_GAP_SEC:
        return None, f"gap_too_large (gap_len={last_gap:.1f}s > {MAX_ALLOWED_GAP_SEC}s)"

    # ---- lấy dòng cuối cùng & kiểm tra NaN ----
    last_row = df.iloc[-1]

    # đảm bảo đủ feature
    missing_feats = [c for c in FEATURE_COLS if c not in last_row.index]
    if missing_feats:
        return None, f"missing_features: {missing_feats}"

    feat_values = last_row[FEATURE_COLS]
    if feat_values.isna().any():
        return None, "nan_in_features"

    feat_df = feat_values.to_frame().T
    return feat_df, "ok"


# ====================== API CHÍNH: is_volatility ======================

def is_volatility(
    lob_data: Union[pd.DataFrame, list, tuple],
    threshold: float = 0.85,
    verbose: bool = True,
) -> bool:
    """
    Hàm tổng:

    Parameters
    ----------
    lob_data : pd.DataFrame | list | tuple
        Dữ liệu LOB gần nhất với cột:
        ts_ms, ts_iso, best_bid, best_ask, last_price,
        best_bid_qty, best_ask_qty, bid_notional_5, ask_notional_5

        len(lob_data) >= MIN_HISTORY_ROWS (~30) để build đủ feature.

    threshold : float
        Ngưỡng p(move_10s) để classify (RF tốt nhất trên VAL ~0.85).

    verbose : bool
        True => in ra trạng thái / debug; False => chỉ trả về True/False.

    Returns
    -------
    bool
        True  => RF dự đoán có "vol move 10s" (p >= threshold)
        False => không move / hoặc không đủ điều kiện predict.
    """

    # 1) Chuẩn hóa input
    df_lob = _prepare_lob_df(lob_data)
    last_ts = df_lob["ts_iso"].iloc[-1]
    last_price = float(df_lob["last_price"].iloc[-1])

    # 2) Build feature
    feat_df, status = _build_feature_from_lob(df_lob)

    if status != "ok":
        if verbose:
            print(
                f"[is_volatility] {last_ts.isoformat()}  last_price={last_price:.4f}  "
                f"status={status} -> RETURN False"
            )
        return False

    # 3) Load model & predict
    model = _load_rf_model()
    proba = float(model.predict_proba(feat_df)[0, 1])
    is_move = proba >= threshold

    if verbose:
        label_text = "MOVE" if is_move else "FLAT"
        print(
            f"[is_volatility] {last_ts.isoformat()}  last_price={last_price:.4f}  "
            f"p(move_10s)={proba:.3f}  threshold={threshold:.2f}  -> {label_text}"
        )

    return bool(is_move)


# ====================== DEMO NHỎ (offline test) ======================

if __name__ == "__main__":
    """
    Demo nhanh: đọc 60 dòng cuối từ file raw 1s của Binance để test.
    Chỉ để anh chạy thử, sau đó có thể xóa / comment phần này.
    """
    RAW_PATH = r"D:\MM\data\CAKEUSDT_binance_lob_fn.csv"

    if os.path.exists(RAW_PATH):
        print(f"[DEMO] Loading raw LOB sample from {RAW_PATH} ...")
        df_raw = pd.read_csv(RAW_PATH)
        # parse ts_iso
        df_raw["ts_iso"] = pd.to_datetime(df_raw["ts_iso"], utc=True)
        df_raw = df_raw.sort_values("ts_iso")

        # lấy 100 dòng cuối để test
        lob_sample = df_raw.iloc[-100:].copy()
        print(f"[DEMO] lob_sample shape: {lob_sample.shape}")

        res = is_volatility(lob_sample, threshold=0.85, verbose=True)
        print(f"[DEMO] is_volatility => {res}")
    else:
        print(f"[DEMO] RAW_PATH không tồn tại: {RAW_PATH}")
