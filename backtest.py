import os
import time
import requests
import ccxt
import numpy as np
import pandas as pd

SYMBOLS = {
    "BTC/USDT:USDT": 15,
    "ETH/USDT:USDT": 15,
    "SOL/USDT:USDT": 10,
    "ADA/USDT:USDT": 10,
}
DAYS = 90
MARGIN_RATIO = 0.25
FEE_RATE = 0.0002  # 지정가(Maker) 편도 0.02%


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


def add_indicators(df):
    # 15분봉 지표
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["prev_low"] = df["low"].shift(1)
    df["prev_high"] = df["high"].shift(1)
    df["prev_close"] = df["close"].shift(1)

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

    # 1시간봉 추세 + ADX (미래 참조 방지 1봉 시프트)
    df_1h = (
        df.set_index("dt")
        .resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
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

    # 4시간봉 대추세 필터 (미래 참조 방지 1봉 시프트)
    df_4h = (
        df.set_index("dt")
        .resample("4h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    df_4h["ema20_4h"] = df_4h["close"].ewm(span=20, adjust=False).mean()
    df_4h["ema50_4h"] = df_4h["close"].ewm(span=50, adjust=False).mean()

    df_1h_s = df_1h[["ema20_1h", "ema50_1h", "adx14_1h"]].shift(1).reset_index()
    df_4h_s = df_4h[["ema20_4h", "ema50_4h"]].shift(1).reset_index()

    df = pd.merge_asof(df.sort_values("dt"), df_1h_s.sort_values("dt"), on="dt", direction="backward")
    df = pd.merge_asof(df.sort_values("dt"), df_4h_s.sort_values("dt"), on="dt", direction="backward")
    return df.dropna().reset_index(drop=True)


def run_backtest(df, symbol, leverage):
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
            notional = pos["margin"] * leverage
            closed = False

            if side == "LONG":
                # 1) 손절 체크
                if row["low"] <= sl:
                    rem = 0.4 if pos["tp1_done"] else 1.0
                    gross = notional * rem * ((sl - entry) / entry)
                    fee = notional * rem * FEE_RATE
                    equity += gross - fee
                    net_pnl = pos["realized_pnl"] + gross - fee
                    trades.append({"symbol": symbol, "dt": row["dt"], "side": side, "pnl": net_pnl, "equity": equity})
                    consecutive_losses = consecutive_losses + 1 if net_pnl < 0 else 0
                    pos = None
                    closed = True
                else:
                    # 2) 1차 익절 (1.8 ATR, 60% 물량) + 본절 이동
                    if not pos["tp1_done"] and row["high"] >= pos["tp1"]:
                        gross = notional * 0.6 * ((pos["tp1"] - entry) / entry)
                        fee = notional * 0.6 * FEE_RATE
                        equity += gross - fee
                        pos["realized_pnl"] += gross - fee
                        pos["tp1_done"] = True
                        pos["sl"] = entry * (1 + FEE_RATE * 2)

                    # 3) 2차 익절 (3.6 ATR 또는 8시간 경과)
                    if pos is not None and pos["tp1_done"]:
                        if row["high"] >= pos["tp2"] or pos["bars"] >= 32:
                            exit_p = pos["tp2"] if row["high"] >= pos["tp2"] else row["close"]
                            gross = notional * 0.4 * ((exit_p - entry) / entry)
                            fee = notional * 0.4 * FEE_RATE
                            equity += gross - fee
                            net_pnl = pos["realized_pnl"] + gross - fee
                            trades.append({"symbol": symbol, "dt": row["dt"], "side": side, "pnl": net_pnl, "equity": equity})
                            consecutive_losses = consecutive_losses + 1 if net_pnl < 0 else 0
                            pos = None
                            closed = True
                    elif pos is not None and pos["bars"] >= 32:
                        gross = notional * ((row["close"] - entry) / entry)
                        fee = notional * FEE_RATE
                        equity += gross - fee
                        net_pnl = pos["realized_pnl"] + gross - fee
                        trades.append({"symbol": symbol, "dt": row["dt"], "side": side, "pnl": net_pnl, "equity": equity})
                        consecutive_losses = consecutive_losses + 1 if net_pnl < 0 else 0
                        pos = None
                        closed = True

            else:  # SHORT
                if row["high"] >= sl:
                    rem = 0.4 if pos["tp1_done"] else 1.0
                    gross = notional * rem * ((entry - sl) / entry)
                    fee = notional * rem * FEE_RATE
                    equity += gross - fee
                    net_pnl = pos["realized_pnl"] + gross - fee
                    trades.append({"symbol": symbol, "dt": row["dt"], "side": side, "pnl": net_pnl, "equity": equity})
                    consecutive_losses = consecutive_losses + 1 if net_pnl < 0 else 0
                    pos = None
                    closed = True
                else:
                    if not pos["tp1_done"] and row["low"] <= pos["tp1"]:
                        gross = notional * 0.6 * ((entry - pos["tp1"]) / entry)
                        fee = notional * 0.6 * FEE_RATE
                        equity += gross - fee
                        pos["realized_pnl"] += gross - fee
                        pos["tp1_done"] = True
                        pos["sl"] = entry * (1 - FEE_RATE * 2)

                    if pos is not None and pos["tp1_done"]:
                        if row["low"] <= pos["tp2"] or pos["bars"] >= 32:
                            exit_p = pos["tp2"] if row["low"] <= pos["tp2"] else row["close"]
                            gross = notional * 0.4 * ((entry - exit_p) / entry)
                            fee = notional * 0.4 * FEE_RATE
                            equity += gross - fee
                            net_pnl = pos["realized_pnl"] + gross - fee
                            trades.append({"symbol": symbol, "dt": row["dt"], "side": side, "pnl": net_pnl, "equity": equity})
                            consecutive_losses = consecutive_losses + 1 if net_pnl < 0 else 0
                            pos = None
                            closed = True
                    elif pos is not None and pos["bars"] >= 32:
                        gross = notional * ((entry - row["close"]) / entry)
                        fee = notional * FEE_RATE
                        equity += gross - fee
                        net_pnl = pos["realized_pnl"] + gross - fee
                        trades.append({"symbol": symbol, "dt": row["dt"], "side": side, "pnl": net_pnl, "equity": equity})
                        consecutive_losses = consecutive_losses + 1 if net_pnl < 0 else 0
                        pos = None
                        closed = True

            if closed:
                peak_equity = max(peak_equity, equity)
                max_dd = max(max_dd, (peak_equity - equity) / peak_equity * 100)
                if (equity - day_start_equity) / day_start_equity >= 0.10 or consecutive_losses >= 2:
                    day_locked = True

        if pos is None and not day_locked and equity > 50:
            if row["atr_pct"] < 0.0025:
                continue

            # 4시간봉 + 1시간봉 이중 정배열/역배열 + ADX >= 22
            long_trend = (
                (row["ema20_4h"] > row["ema50_4h"])
                and (row["ema20_1h"] > row["ema50_1h"])
                and (row["adx14_1h"] >= 22)
            )
            short_trend = (
                (row["ema20_4h"] < row["ema50_4h"])
                and (row["ema20_1h"] < row["ema50_1h"])
                and (row["adx14_1h"] >= 22)
            )

            # EMA20 눌림 후 RSI 반전 + 직전 고점(저점) 돌파 확인 캔들
            long_sig = (
                long_trend
                and (row["prev_low"] <= row["ema20"])
                and (row["prev_rsi"] <= 45)
                and (row["rsi14"] > row["prev_rsi"])
                and (row["close"] > row["prev_high"])
            )
            short_sig = (
                short_trend
                and (row["prev_high"] >= row["ema20"])
                and (row["prev_rsi"] >= 55)
                and (row["rsi14"] < row["prev_rsi"])
                and (row["close"] < row["prev_low"])
            )

            if long_sig or short_sig:
                side = "LONG" if long_sig else "SHORT"
                entry = row["close"]
                atr = row["atr14"]
                margin = equity * MARGIN_RATIO
                entry_fee = margin * leverage * FEE_RATE
                equity -= entry_fee

                if side == "LONG":
                    sl = entry - 2.2 * atr
                    tp1 = entry + 1.8 * atr
                    tp2 = entry + 3.6 * atr
                else:
                    sl = entry + 2.2 * atr
                    tp1 = entry - 1.8 * atr
                    tp2 = entry - 3.6 * atr

                pos = {
                    "side": side,
                    "entry": entry,
                    "sl": sl,
                    "tp1": tp1,
                    "tp2": tp2,
                    "tp1_done": False,
                    "margin": margin,
                    "realized_pnl": -entry_fee,
                    "bars": 0,
                }

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        return {"symbol": symbol, "trades": 0, "win_rate": 0.0, "ret_pct": 0.0, "mdd_pct": 0.0}, tdf

    wins = (tdf["pnl"] > 0).sum()
    win_rate = wins / len(tdf) * 100
    ret_pct = (equity - 1000.0) / 1000.0 * 100
    return {
        "symbol": symbol,
        "trades": len(tdf),
        "win_rate": round(win_rate, 2),
        "ret_pct": round(ret_pct, 2),
        "mdd_pct": round(max_dd, 2),
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
    summaries = []
    all_trades = []

    for sym, lev in SYMBOLS.items():
        raw = fetch_15m_ohlcv(exchange, sym, days=DAYS)
        df = add_indicators(raw)
        res, tdf = run_backtest(df, sym, lev)
        summaries.append(res)
        if not tdf.empty:
            all_trades.append(tdf)

    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv("backtest_trades.csv", index=False)

    lines = [f"📊 이중추세+분할익절 백테스트 결과 ({DAYS}일)"]
    for s in summaries:
        lines.append(
            f"- {s['symbol'].split('/')[0]}: 거래 {s['trades']}회 | 승률 {s['win_rate']}% | 누적 {s['ret_pct']}% | MDD {s['mdd_pct']}%"
        )
    msg = "\n".join(lines)
    print(msg)
    send_telegram(msg)


if __name__ == "__main__":
    main()
