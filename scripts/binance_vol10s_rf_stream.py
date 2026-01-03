# D:\MM\binance_vol10s_rf_stream.py
"""
Streaming orderbook depth5 từ Binance cho CAKEUSDT,
mỗi ~1s snapshot + predict p(move_10s) dùng RandomForest train offline.

- Model: D:\MM\models_vol\rf_vol_10s.pkl
- Ghi CSV: ts_ms, ts_iso, best_bid, best_ask, last_price(mid), ..., proba_rf, label_hat_rf
- In realtime trên terminal:
  last_price(mid), mid, spread, rv_10s, range_10s_cur, gap_len, has_tick, p(move_10s), label_hat (MOVE/FLAT)
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import websockets
from joblib import load


# ========================== CONFIG ==========================

# Single stream depth5 cho CAKEUSDT
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/cakeusdt@depth5@100ms"

SYMBOL = "CAKEUSDT"

OUTPUT_DIR = r"D:\MM\data"
MODEL_PATH_RF = r"D:\MM\models_vol\rf_vol_10s.pkl"

SNAPSHOT_INTERVAL_SEC = 1.0      # mỗi 1s in/predict 1 lần
HISTORY_SEC = 600                # ~10 phút history
PRED_THRESHOLD_RF = 0.85         # threshold từ val cho RandomForest

FEATURE_COLS_RF = [
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

MIN_HISTORY_FOR_PRED = 30  # cần tối thiểu ~30s history


# ========================== UTIL ==========================

def _get_output_file(symbol: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol}_binance_vol10s_stream_{ts_str}.csv"
    return os.path.join(OUTPUT_DIR, filename)


def _compute_top_and_notional_5(levels: List[List[str]]) -> (Optional[float], Optional[float], float):
    """
    levels: [[price, qty, ...], ...] (string) từ Binance depth.
    Trả về: (top_price, top_qty, notional_5) – ở đây notional_5 là sum(price * qty) top 5.
    """
    top_price = None
    top_qty = None
    notional_5 = 0.0

    if levels and len(levels[0]) >= 2:
        try:
            top_price = float(levels[0][0])
            top_qty = float(levels[0][1])
        except Exception:
            top_price = None
            top_qty = None

    for lvl in levels[:5]:
        if len(lvl) < 2:
            continue
        p_str, q_str = lvl[0], lvl[1]
        try:
            p = float(p_str)
            q = float(q_str)
            notional_5 += p * q
        except Exception:
            continue

    return top_price, top_qty, notional_5


# ========================== SHARED STATE ==========================

@dataclass
class SharedState:
    """
    Chỉ cần depth stream, không cần ticker nữa.
    """
    last_depth: Optional[Dict[str, Any]] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def update_depth(self, d: Dict[str, Any]) -> None:
        async with self.lock:
            self.last_depth = d

    async def get_snapshot_raw(self) -> Optional[Dict[str, Any]]:
        async with self.lock:
            if self.last_depth is None:
                return None
            return self.last_depth.copy()


@dataclass
class Snapshot:
    ts_ms: int
    ts_iso: datetime
    best_bid: float
    best_ask: float
    last_price: float    # ở đây dùng mid làm proxy last_price
    best_bid_qty: float
    best_ask_qty: float
    bid_notional_5: float
    ask_notional_5: float
    has_tick: int
    gap_len: float


# ========================== FEATURE STATE ==========================

class FeatureState:
    """
    Giữ history (DataFrame) + tính feature giống offline:
    mid, spread, ret_*, rv_10s, range_10s_cur, depth_*, qty_*, has_tick, gap_len, tod_sin/cos.
    """

    def __init__(self, feature_cols: List[str], history_sec: int = HISTORY_SEC):
        self.feature_cols = feature_cols
        self.history_sec = history_sec
        self.df: pd.DataFrame = pd.DataFrame()

    def update_and_get_features(self, snapshot: Snapshot) -> Optional[pd.DataFrame]:
        ts = snapshot.ts_iso

        mid = 0.5 * (snapshot.best_bid + snapshot.best_ask)
        spread = snapshot.best_ask - snapshot.best_bid

        depth_bid = snapshot.bid_notional_5
        depth_ask = snapshot.ask_notional_5
        depth_sum = depth_bid + depth_ask
        denom_depth = depth_sum if depth_sum != 0 else np.nan
        depth_imb = (depth_bid - depth_ask) / denom_depth if not math.isclose(denom_depth, 0.0) else 0.0

        qty_bid = snapshot.best_bid_qty
        qty_ask = snapshot.best_ask_qty
        qty_sum = qty_bid + qty_ask
        denom_qty = qty_sum if qty_sum != 0 else np.nan
        qty_imb = (qty_bid - qty_ask) / denom_qty if not math.isclose(denom_qty, 0.0) else 0.0

        # time-of-day
        tod = ts.time()
        tod_sec = tod.hour * 3600 + tod.minute * 60 + tod.second
        angle = 2 * math.pi * tod_sec / 86400.0
        tod_sin = math.sin(angle)
        tod_cos = math.cos(angle)

        base_row = {
            "mid": mid,
            "spread": spread,
            "depth_bid": depth_bid,
            "depth_ask": depth_ask,
            "depth_sum": depth_sum,
            "depth_imb": depth_imb,
            "qty_bid": qty_bid,
            "qty_ask": qty_ask,
            "qty_sum": qty_sum,
            "qty_imb": qty_imb,
            "has_tick": snapshot.has_tick,
            "gap_len": snapshot.gap_len,
            "tod_sin": tod_sin,
            "tod_cos": tod_cos,
        }

        if self.df.empty:
            self.df = pd.DataFrame([base_row], index=[ts])
        else:
            for k in base_row.keys():
                if k not in self.df.columns:
                    self.df[k] = np.nan
            self.df.loc[ts] = base_row

        # giữ lịch sử gần nhất
        self.df = self.df.sort_index()
        if len(self.df) > self.history_sec:
            self.df = self.df.iloc[-self.history_sec:]

        df = self.df

        # log_mid + returns
        df["log_mid"] = np.log(df["mid"].replace(0, np.nan))
        df["ret_1s"] = df["log_mid"].diff()
        df["ret_2s"] = df["log_mid"] - df["log_mid"].shift(2)
        df["ret_5s"] = df["log_mid"] - df["log_mid"].shift(5)

        # realized vol 10s (std của ret_1s trong 10s)
        df["rv_10s"] = df["ret_1s"].rolling(window=10, min_periods=5).std()

        # range_10s_cur: high-low mid trong 10s gần nhất
        roll_max = df["mid"].rolling(window=10, min_periods=5).max()
        roll_min = df["mid"].rolling(window=10, min_periods=5).min()
        df["range_10s_cur"] = roll_max - roll_min

        self.df = df

        if len(df) < MIN_HISTORY_FOR_PRED:
            return None

        last_row = df.iloc[-1]

        # check đủ cột + không NaN
        if any(col not in last_row.index for col in self.feature_cols):
            return None
        feat_values = last_row[self.feature_cols]
        if feat_values.isna().any():
            return None

        return feat_values.to_frame().T


# ========================== WS LOOP (BINANCE depth5 ONLY) ==========================

async def ws_loop(state: SharedState) -> None:
    reconnect_count = 0

    while True:
        try:
            print(f"[INFO] Connecting to Binance WS {BINANCE_WS_URL} (attempt {reconnect_count + 1})...")
            async with websockets.connect(
                BINANCE_WS_URL,
                ping_interval=20,
                ping_timeout=10,
            ) as websocket:
                reconnect_count = 0
                print("[INFO] Connected to Binance depth5 stream for CAKEUSDT")

                while True:
                    msg = await websocket.recv()
                    data = json.loads(msg)

                    # Single-stream /ws -> data là depth update, có các field:
                    # e: event type, E: event time (ms), s: symbol, b: bids, a: asks, ...
                    if data.get("s", "").upper() != SYMBOL:
                        continue

                    depth_msg = {
                        "event_time": data.get("E"),
                        "bids": data.get("b", []),
                        "asks": data.get("a", []),
                    }
                    await state.update_depth(depth_msg)

        except KeyboardInterrupt:
            print("[INFO] ws_loop stopped by user.")
            raise

        except websockets.exceptions.ConnectionClosed as e:
            reconnect_count += 1
            print(f"[WARN] WebSocket connection closed: {e}. Reconnecting in 3s...")
            await asyncio.sleep(3)

        except Exception as e:
            reconnect_count += 1
            print(f"[ERROR] ws_loop error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)


# ========================== SNAPSHOT & PREDICT LOOP ==========================

async def snapshot_and_predict_loop(
    state: SharedState,
    feat_state: FeatureState,
    model_rf,
    output_file: str,
) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    last_event_time: Optional[int] = None
    last_tick_walltime: Optional[datetime] = None

    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "ts_ms",
                "ts_iso",
                "best_bid",
                "best_ask",
                "last_price",  # mid làm proxy
                "best_bid_qty",
                "best_ask_qty",
                "bid_notional_5",
                "ask_notional_5",
                "has_tick",
                "gap_len",
                "mid",
                "spread",
                "rv_10s",
                "range_10s_cur",
                "proba_rf",
                "label_hat_rf",
            ]
        )

        print(f"[INFO] Snapshot loop started. Writing to: {output_file}")

        while True:
            try:
                raw = await state.get_snapshot_raw()
                now = datetime.now(timezone.utc)

                if raw is None:
                    print(f"[PRED] {now.isoformat()}  warming up (chưa có depth)...")
                    await asyncio.sleep(SNAPSHOT_INTERVAL_SEC)
                    continue

                depth_E = raw.get("event_time")
                bids = raw.get("bids", [])
                asks = raw.get("asks", [])

                if depth_E is None:
                    print(f"[PRED] {now.isoformat()}  warming up (thiếu event_time)...")
                    await asyncio.sleep(SNAPSHOT_INTERVAL_SEC)
                    continue

                # has_tick + gap_len theo event_time
                if last_event_time is None or depth_E != last_event_time:
                    has_tick = 1
                    last_event_time = depth_E
                    last_tick_walltime = now
                    gap_len = 0.0
                else:
                    has_tick = 0
                    if last_tick_walltime is not None:
                        gap_len = (now - last_tick_walltime).total_seconds()
                    else:
                        gap_len = 0.0

                try:
                    ts_ms = int(depth_E)
                except Exception:
                    ts_ms = int(now.timestamp() * 1000)
                ts_iso = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)

                best_bid, best_bid_qty, bid_notional_5 = _compute_top_and_notional_5(bids)
                best_ask, best_ask_qty, ask_notional_5 = _compute_top_and_notional_5(asks)

                if best_bid is None or best_ask is None:
                    print(f"[PRED] {ts_iso.isoformat()}  warming up (thiếu best_bid/ask)...")
                    await asyncio.sleep(SNAPSHOT_INTERVAL_SEC)
                    continue

                mid = 0.5 * (best_bid + best_ask)
                last_price_val = mid  # proxy

                snap = Snapshot(
                    ts_ms=ts_ms,
                    ts_iso=ts_iso,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    last_price=last_price_val,
                    best_bid_qty=best_bid_qty or 0.0,
                    best_ask_qty=best_ask_qty or 0.0,
                    bid_notional_5=bid_notional_5,
                    ask_notional_5=ask_notional_5,
                    has_tick=has_tick,
                    gap_len=gap_len,
                )

                feat_df = feat_state.update_and_get_features(snap)

                rv_10s = np.nan
                range_10s_cur = np.nan
                if feat_state.df.shape[0] > 0:
                    last_row = feat_state.df.iloc[-1]
                    rv_10s = float(last_row.get("rv_10s", np.nan))
                    range_10s_cur = float(last_row.get("range_10s_cur", np.nan))

                proba_rf = np.nan
                label_hat_rf = 0

                if feat_df is None:
                    print(
                        f"[PRED] {snap.ts_iso.isoformat()}  "
                        f"last_price(mid)={last_price_val:.4f} mid={mid:.4f} spread={snap.best_ask - snap.best_bid:.4f}  "
                        f"rv_10s={rv_10s:.6f} range_10s={range_10s_cur:.4f}  "
                        f"gap_len={gap_len:.1f}s has_tick={has_tick}  "
                        f"p(move_10s)=warming_up"
                    )
                else:
                    proba_rf = float(model_rf.predict_proba(feat_df)[0, 1])
                    label_hat_rf = int(proba_rf >= PRED_THRESHOLD_RF)
                    label_text = "MOVE" if label_hat_rf == 1 else "FLAT"

                    print(
                        f"[PRED] {snap.ts_iso.isoformat()}  "
                        f"last_price(mid)={last_price_val:.4f} mid={mid:.4f} spread={snap.best_ask - snap.best_bid:.4f}  "
                        f"rv_10s={rv_10s:.6f} range_10s={range_10s_cur:.4f}  "
                        f"gap_len={gap_len:.1f}s has_tick={has_tick}  "
                        f"p(move_10s)={proba_rf:.3f}  label_hat={label_hat_rf} ({label_text})"
                    )

                # ghi CSV
                writer.writerow(
                    [
                        snap.ts_ms,
                        snap.ts_iso.isoformat(),
                        snap.best_bid,
                        snap.best_ask,
                        last_price_val,
                        snap.best_bid_qty,
                        snap.best_ask_qty,
                        snap.bid_notional_5,
                        snap.ask_notional_5,
                        has_tick,
                        gap_len,
                        mid,
                        snap.best_ask - snap.best_bid,
                        rv_10s,
                        range_10s_cur,
                        proba_rf,
                        label_hat_rf,
                    ]
                )
                f.flush()

                await asyncio.sleep(SNAPSHOT_INTERVAL_SEC)

            except KeyboardInterrupt:
                print("\n[INFO] Snapshot loop stopped by user.")
                break

            except Exception as e:
                print(f"[ERROR] snapshot_loop error: {e}")
                await asyncio.sleep(1.0)


# ========================== MAIN ==========================

def load_rf_model():
    print(f"[INFO] Loading RF vol model from: {MODEL_PATH_RF}")
    model = load(MODEL_PATH_RF)
    print("[INFO] RF model loaded.")
    return model


async def main():
    rf_model = load_rf_model()
    state = SharedState()
    feat_state = FeatureState(feature_cols=FEATURE_COLS_RF, history_sec=HISTORY_SEC)
    output_file = _get_output_file(SYMBOL)
    print(f"[INFO] Output CSV: {output_file}")

    ws_task = asyncio.create_task(ws_loop(state))
    snap_task = asyncio.create_task(snapshot_and_predict_loop(state, feat_state, rf_model, output_file))

    await asyncio.gather(ws_task, snap_task)


if __name__ == "__main__":
    asyncio.run(main())
