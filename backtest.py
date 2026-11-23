import os
import time
import requests
import ccxt
import numpy as np
import pandas as pd

SYMBOLS = {
    "BTC/USDT:USDT": 12,
    "ETH/USDT:USDT": 12,
    "SOL/USDT:USDT": 8,
    "ADA/USDT:USDT": 8,
}
DAYS = 90
MARGIN_RATIO = 0.25
FEE_RATE = 0.0005  # 편도 0.05% (왕복 0.10% 보수적 반영)


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
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["high20"] = df["high"].rolling(20).max().shift(1)
    df["low20"] = df["low"].rolling(20).min().shift(1)

    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))
    df["prev_rsi"] = df["rsi14"].shift(1)

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
    tr = pd.concat(
        [
            df_1h["high"] - df_1h["low"],
            (df_1h["high"] - df_1h["close"].shift(1)).abs(),
            (df_1h["low"] - df_1h["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df_1h.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df_1h.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df_1h["adx14_1h"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    df_1h_shifted = df_1h[["ema20_1h", "ema50_1h", "adx14_1h"]].shift(1).reset_index()
    df = pd.merge_asof(df.sort_values("dt"), df_1h_shifted.sort_values("dt"), on="dt", direction="backward")
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
            side = pos["side"]
            entry = pos["entry"]
            sl = pos["sl"]
            notional = pos["margin"] * leverage

            if side == "LONG":
                if row["low"] <= sl:
                    rem = 0.5 if pos["tp1_done"] else 1.0
                    pnl_pct = (sl - entry) / entry
                    pnl = notional * rem * pnl_pct - notional * rem * FEE_RATE
                    equity += pnl
                    total_pnl = pos["realized_pnl"] + pnl
                    trades.append({"symbol": symbol, "dt": row["dt"], "side": side, "pnl": total_pnl, "equity": equity})
                    consecutive_losses = consecutive_losses + 1 if total_pnl < 0 else 0
                    pos = None
                else:
                    if not pos["tp1_done"] and row["high"] >= entry * 1.02:
                        pnl = notional * 0.5 * 0.02 - notional * 0.5 * FEE_RATE
                        equity += pnl
                        pos["realized_pnl"] += pnl
                        pos["tp1_done"] = True
                        pos["sl"] = entry
                    if pos is not None and pos["tp1_done"]:
                        if row["high"] >= entry * 1.035 or row["close"] < row["ema20"]:
                            exit_p = entry * 1.035 if row["high"] >= entry * 1.035 else row["close"]
                            pnl = notional * 0.5 * ((exit_p - entry) / entry) - notional * 0.5 * FEE_RATE
                            equity += pnl
                            total_pnl = pos["realized_pnl"] + pnl
                            trades.append({"symbol": symbol, "dt": row["dt"], "side": side, "pnl": total_pnl, "equity": equity})
                            consecutive_losses = 0
                            pos = None

            elif side == "SHORT":
                if row["high"] >= sl:
                    rem = 0.5 if pos["tp1_done"] else 1.0
                    pnl_pct = (entry - sl) / entry
                    pnl = notional * rem * pnl_pct - notional * rem * FEE_RATE
                    equity += pnl
                    total_pnl = pos["realized_pnl"] + pnl
                    trades.append({"symbol": symbol, "dt": row["dt"], "side": side, "pnl": total_pnl, "equity": equity})
                    consecutive_losses = consecutive_losses + 1 if total_pnl < 0 else 0
                    pos = None
                else:
                    if not pos["tp1_done"] and row["low"] <= entry * 0.98:
                        pnl = notional * 0.5 * 0.02 - notional * 0.5 * FEE_RATE
                        equity += pnl
                        pos["realized_pnl"] += pnl
                        pos["tp1_done"] = True
                        pos["sl"] = entry
                    if pos is not None and pos["tp1_done"]:
                        if row["low"] <= entry * 0.965 or row["close"] > row["ema20"]:
                            exit_p = entry * 0.965 if row["low"] <= entry * 0.965 else row["close"]
                            pnl = notional * 0.5 * ((entry - exit_p) / entry) - notional * 0.5 * FEE_RATE
                            equity += pnl
                            total_pnl = pos["realized_pnl"] + pnl
                            trades.append({"symbol": symbol, "dt": row["dt"], "side": side, "pnl": total_pnl, "equity": equity})
                            consecutive_losses = 0
                            pos = None

            peak_equity = max(peak_equity, equity)
            max_dd = max(max_dd, (peak_equity - equity) / peak_equity * 100)

            if (equity - day_start_equity) / day_start_equity >= 0.10 or consecutive_losses >= 2:
                day_locked = True

        if pos is None and not day_locked and equity > 50:
            adx_ok = row["adx14_1h"] >= 20
            long_trend = adx_ok and (row["ema20_1h"] > row["ema50_1h"])
            short_trend = adx_ok and (row["ema20_1h"] < row["ema50_1h"])

            long_sig = long_trend and (
                (row["close"] > row["high20"] and row["volume"] >= 1.5 * row["vol_ma20"])
                or (row["low"] <= row["ema20"] * 1.002 and row["prev_rsi"] < 45 and row["rsi14"] >= 45)
            )
            short_sig = short_trend and (
                (row["close"] < row["low20"] and row["volume"] >= 1.5 * row["vol_ma20"])
                or (row["high"] >= row["ema20"] * 0.998 and row["prev_rsi"] > 55 and row["rsi14"] <= 55)
            )

            if long_sig or short_sig:
                side = "LONG" if long_sig else "SHORT"
                entry = row["close"]
                margin = equity * MARGIN_RATIO
                equity -= margin * leverage * FEE_RATE
                sl = entry * (0.988 if side == "LONG" else 1.012)
                pos = {
                    "side": side,
                    "entry": entry,
                    "sl": sl,
                    "tp1_done": False,
                    "margin": margin,
                    "realized_pnl": -margin * leverage * FEE_RATE,
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

    lines = [f"📊 최근 {DAYS}일 레버리지 전략 백테스트 결과"]
    for s in summaries:
        lines.append(
            f"- {s['symbol'].split('/')[0]}: 거래 {s['trades']}회 | 승률 {s['win_rate']}% | 누적 {s['ret_pct']}% | MDD {s['mdd_pct']}%"
        )
    msg = "\n".join(lines)
    print(msg)
    send_telegram(msg)


if __name__ == "__main__":
    main()

