import os
import time
import requests
import ccxt
import numpy as np
import pandas as pd

MAX_LEVERAGE = {
    "BTC/USDT:USDT": 15,
    "ETH/USDT:USDT": 15,
    "SOL/USDT:USDT": 10,
    "ADA/USDT:USDT": 10,
}
DAYS = 90
RISK_PER_TRADE = 0.045  # 1회 손절 시 계좌 손실 한도 4.5% (2연속 익절 시 약 +8.5%)
FEE_RATE = 0.0002       # 바이비트 지정가(Maker) 편도 0.02%


def fetch_15m_ohlcv(exchange, symbol, days=90):
    since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
    all_rows = []
    while True:
        rows = exchange.fetch_ohlcv(symbol, timeframe="15m", since=since, limit=100)
        if not rows:
            break
        all_rows.extend(rows)
        last_ts = rows[-1][0]
        if last_ts <= since or len(rows) < 100:
            break
        since = last_ts + 15 * 60 * 1000
        time.sleep(0.15)

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def build_htf_features(df):
    # 1시간봉 추세 및 ADX (미래 참조 방지 1봉 시프트)
    df_1h = (
        df.set_index("dt")
        .resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    df_1h["close_1h"] = df_1h["close"]
    df_1h["ema20_1h"] = df_1h["close"].ewm(span=20, adjust=False).mean()
    df_1h["ema50_1h"] = df_1h["close"].ewm(span=50, adjust=False).mean()

    up_move = df_1h["high"].diff()
    down_move = -df_1h["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1h = pd.concat(
        [
            df_1h["high"] - df_1h["low"],
            (df_1h["high"] - df_1h["close"].shift(1)).abs(),
            (df_1h["low"] - df_1h["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr1h = tr1h.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df_1h.index).ewm(alpha=1 / 14, adjust=False).mean() / atr1h
    minus_di = 100 * pd.Series(minus_dm, index=df_1h.index).ewm(alpha=1 / 14, adjust=False).mean() / atr1h
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df_1h["adx14_1h"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # 4시간봉 대추세 및 기울기 (미래 참조 방지 1봉 시프트)
    df_4h = (
        df.set_index("dt")
        .resample("4h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    df_4h["close_4h"] = df_4h["close"]
    df_4h["ema20_4h"] = df_4h["close"].ewm(span=20, adjust=False).mean()
    df_4h["ema20_4h_prev"] = df_4h["ema20_4h"].shift(1)
    df_4h["ema50_4h"] = df_4h["close"].ewm(span=50, adjust=False).mean()

    s_1h = df_1h[["close_1h", "ema20_1h", "ema50_1h", "adx14_1h"]].shift(1).reset_index()
    s_4h = df_4h[["close_4h", "ema20_4h", "ema20_4h_prev", "ema50_4h"]].shift(1).reset_index()
    return s_1h, s_4h


def add_indicators(df, btc_1h, btc_4h):
    df["bb_mid"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2.0 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2.0 * bb_std
    df["prev_bb_upper"] = df["bb_upper"].shift(1)
    df["prev_bb_lower"] = df["bb_lower"].shift(1)
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))
    df["prev_rsi"] = df["rsi14"].shift(1)

    tr15 = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr15.ewm(alpha=1 / 14, adjust=False).mean()
    df["atr_pct"] = df["atr14"] / df["close"]

    s_1h, s_4h = build_htf_features(df)
    df = pd.merge_asof(df.sort_values("dt"), s_1h.sort_values("dt"), on="dt", direction="backward")
    df = pd.merge_asof(df.sort_values("dt"), s_4h.sort_values("dt"), on="dt", direction="backward")
    df = pd.merge_asof(df.sort_values("dt"), btc_1h.sort_values("dt"), on="dt", direction="backward")
    df = pd.merge_asof(df.sort_values("dt"), btc_4h.sort_values("dt"), on="dt", direction="backward")
    return df.dropna().reset_index(drop=True)


def run_backtest(df, symbol, max_lev):
    equity = 1000.0
    peak_equity = equity
    max_dd = 0.0

    pos = None
    trades = []

    current_day = None
    day_start_equity = equity
    consecutive_losses = 0
    day_locked = False

    for _, row in df.iterrows():
        day = row["dt"].date()
        if day != current_day:
            current_day = day
            day_start_equity = equity
            consecutive_losses = 0
            day_locked = False

        if pos is not None:
            pos["bars"] += 1
            side = pos["side"]
            entry = pos["entry"]
            sl = pos["sl"]
            tp = pos["tp"]
            notional = pos["notional"]
            exit_price = None

            if side == "LONG":
                if row["low"] <= sl:
                    exit_price = sl
                elif row["high"] >= tp:
                    exit_price = tp
                elif pos["bars"] >= 144:  # 최대 36시간 보유
                    exit_price = row["close"]
            else:
                if row["high"] >= sl:
                    exit_price = sl
                elif row["low"] <= tp:
                    exit_price = tp
                elif pos["bars"] >= 144:
                    exit_price = row["close"]

            if exit_price is not None:
                pnl_pct = (exit_price - entry) / entry if side == "LONG" else (entry - exit_price) / entry
                gross_pnl = notional * pnl_pct
                exit_fee = notional * FEE_RATE
                net_pnl = gross_pnl - exit_fee - pos["entry_fee"]
                equity += gross_pnl - exit_fee

                trades.append({"symbol": symbol, "dt": row["dt"], "side": side, "pnl": net_pnl, "equity": equity})
                consecutive_losses = consecutive_losses + 1 if net_pnl < 0 else 0
                pos = None

                peak_equity = max(peak_equity, equity)
                max_dd = max(max_dd, (peak_equity - equity) / peak_equity * 100)

                if (equity - day_start_equity) / day_start_equity >= 0.10 or consecutive_losses >= 2:
                    day_locked = True

        if pos is None and not day_locked and equity > 50:
            if row["atr_pct"] < 0.0025:
                continue

            # BTC 4H+1H 추세 동조 & 개별코인 4H 기울기+1H 추세 & ADX >= 22
            long_trend = (
                (row["btc_close_4h"] > row["btc_ema50_4h"])
                and (row["btc_close_1h"] > row["btc_ema20_1h"])
                and (row["ema20_4h"] > row["ema50_4h"])
                and (row["ema20_4h"] > row["ema20_4h_prev"])
                and (row["ema20_1h"] > row["ema50_1h"])
                and (row["adx14_1h"] >= 22)
            )
            short_trend = (
                (row["btc_close_4h"] < row["btc_ema50_4h"])
                and (row["btc_close_1h"] < row["btc_ema20_1h"])
                and (row["ema20_4h"] < row["ema50_4h"])
                and (row["ema20_4h"] < row["ema20_4h_prev"])
                and (row["ema20_1h"] < row["ema50_1h"])
                and (row["adx14_1h"] >= 22)
            )

            # 볼린저밴드 외곽 터치 + RSI 반전 + 거래량 동반 반전 캔들
            long_touch = (row["low"] <= row["bb_lower"]) or (row["prev_low"] <= row["prev_bb_lower"])
            long_rsi = (row["prev_rsi"] <= 36) and (row["rsi14"] > row["prev_rsi"]) and (row["rsi14"] <= 45)
            long_sig = long_trend and long_touch and long_rsi and (row["close"] > row["open"]) and (row["volume"] >= row["vol_ma20"])

            short_touch = (row["high"] >= row["bb_upper"]) or (row["prev_high"] >= row["prev_bb_upper"])
            short_rsi = (row["prev_rsi"] >= 64) and (row["rsi14"] < row["prev_rsi"]) and (row["rsi14"] >= 55)
            short_sig = short_trend and short_touch and short_rsi and (row["close"] < row["open"]) and (row["volume"] >= row["vol_ma20"])

            if long_sig or short_sig:
                side = "LONG" if long_sig else "SHORT"
                entry = row["close"]
                atr = row["atr14"]

                sl_dist = 2.8 * atr
                tp_dist = 2.6 * atr
                sl_pct = sl_dist / entry

                # 동일 리스크 포지션 사이징 (최대 허용 레버리지 한도 내)
                notional = min((equity * RISK_PER_TRADE) / sl_pct, equity * max_lev)
                entry_fee = notional * FEE_RATE
                equity -= entry_fee

                sl = entry - sl_dist if side == "LONG" else entry + sl_dist
                tp = entry + tp_dist if side == "LONG" else entry - tp_dist

                pos = {
                    "side": side,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "notional": notional,
                    "entry_fee": entry_fee,
                    "bars": 0,
                }

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        return {"symbol": symbol, "trades": 0, "win_rate": 0.0, "ret_pct": 0.0, "mdd_pct": 0.0, "pf": 0.0}, tdf

    wins = (tdf["pnl"] > 0).sum()
    win_rate = wins / len(tdf) * 100
    ret_pct = (equity - 1000.0) / 1000.0 * 100
    gross_win = tdf.loc[tdf["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum())
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 99.0

    return {
        "symbol": symbol,
        "trades": len(tdf),
        "win_rate": round(win_rate, 2),
        "ret_pct": round(ret_pct, 2),
        "mdd_pct": round(max_dd, 2),
        "pf": pf,
    }, tdf


def send_telegram(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram secrets not found.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=15)


def main():
    exchange = ccxt.okx({"enableRateLimit": True})
    raw_data = {}
    for sym in MAX_LEVERAGE:
        raw_data[sym] = fetch_15m_ohlcv(exchange, sym, days=DAYS)

    btc_1h_raw, btc_4h_raw = build_htf_features(raw_data["BTC/USDT:USDT"])
    btc_1h = btc_1h_raw[["dt", "close_1h", "ema20_1h"]].rename(
        columns={"close_1h": "btc_close_1h", "ema20_1h": "btc_ema20_1h"}
    )
    btc_4h = btc_4h_raw[["dt", "close_4h", "ema50_4h"]].rename(
        columns={"close_4h": "btc_close_4h", "ema50_4h": "btc_ema50_4h"}
    )

    summaries = []
    all_trades = []
    for sym, lev in MAX_LEVERAGE.items():
        df = add_indicators(raw_data[sym].copy(), btc_1h, btc_4h)
        res, tdf = run_backtest(df, sym, lev)
        summaries.append(res)
        if not tdf.empty:
            all_trades.append(tdf)

    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv("backtest_trades.csv", index=False)

    lines = [f"📊 리스크사이징+BTC동조 고도화 결과 ({DAYS}일)"]
    for s in summaries:
        lines.append(
            f"- {s['symbol'].split('/')[0]}: {s['trades']}회 | 승률 {s['win_rate']}% | 누적 {s['ret_pct']}% | MDD {s['mdd_pct']}% | PF {s['pf']}"
        )
    msg = "\n".join(lines)
    print(msg)
    send_telegram(msg)


if __name__ == "__main__":
    main()
