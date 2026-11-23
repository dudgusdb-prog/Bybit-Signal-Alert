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
RISK_PER_TRADE = 0.065  # 1회 손절 시 계좌 손실 한도 6.5% (익절 시 약 +6.0%)
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

    df_4h = (
        df.set_index("dt")
        .resample("4h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    df_4h["close_4h"] = df_4h["close"]
    df_4h["ema20_4h"] = df_4h["close"].ewm(span=20, adjust=False).mean()
    df_4h["ema50_4h"] = df_4h["close"].ewm(span=50, adjust=False).mean()

    s_1h = df_1h[["ema20_1h", "ema50_1h", "adx14_1h"]].shift(1).reset_index()
    s_4h = df_4h[["close_4h", "ema20_4h", "ema50_4h"]].shift(1).reset_index()
    return s_1h, s_4h


def add_indicators(df, btc_4h):
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["bb_mid"] = df["close"].rolling(20).mean()
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

    s_1h, s_4h = build_htf_features(df)
    df = pd.merge_asof(df.sort_values("dt"), s_1h.sort_values("dt"), on="dt", direction="backward")
    df = pd.merge_asof(df.sort_values("dt"), s_4h.sort_values("dt"), on="dt", direction="backward")
    df = pd.merge_asof(df.sort_values("dt"), btc_4h.sort_values("dt"), on="dt", direction="backward")
    return df.dropna().reset_index(drop=True)


def check_signal(row):
    if row["atr_pct"] < 0.0022:
        return None

    # BTC 4H 대추세 동조 + 개별코인 4H/1H 정배열(역배열) + ADX >= 18
    long_trend = (
        (row["btc_close_4h"] > row["btc_ema50_4h"])
        and (row["ema20_4h"] > row["ema50_4h"])
        and (row["ema20_1h"] > row["ema50_1h"])
        and (row["adx14_1h"] >= 18)
    )
    short_trend = (
        (row["btc_close_4h"] < row["btc_ema50_4h"])
        and (row["ema20_4h"] < row["ema50_4h"])
        and (row["ema20_1h"] < row["ema50_1h"])
        and (row["adx14_1h"] >= 18)
    )

    # 15분봉 EMA20 / 중심선 눌림 후 RSI 반전 양봉(음봉)
    long_pullback = (row["low"] <= row["ema20"]) or (row["prev_low"] <= row["bb_mid"])
    long_rsi = (row["prev_rsi"] <= 48) and (row["rsi14"] > row["prev_rsi"]) and (row["rsi14"] <= 58)
    if long_trend and long_pullback and long_rsi and (row["close"] > row["open"]) and (row["close"] > row["ema20"]):
        return "LONG"

    short_pullback = (row["high"] >= row["ema20"]) or (row["prev_high"] >= row["bb_mid"])
    short_rsi = (row["prev_rsi"] >= 52) and (row["rsi14"] < row["prev_rsi"]) and (row["rsi14"] >= 42)
    if short_trend and short_pullback and short_rsi and (row["close"] < row["open"]) and (row["close"] < row["ema20"]):
        return "SHORT"

    return None


def run_single_backtest(df, symbol, max_lev):
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
            exit_price = None

            if side == "LONG":
                if row["low"] <= pos["sl"]:
                    exit_price = pos["sl"]
                elif row["high"] >= pos["tp"]:
                    exit_price = pos["tp"]
                elif pos["bars"] >= 128:
                    exit_price = row["close"]
            else:
                if row["high"] >= pos["sl"]:
                    exit_price = pos["sl"]
                elif row["low"] <= pos["tp"]:
                    exit_price = pos["tp"]
                elif pos["bars"] >= 128:
                    exit_price = row["close"]

            if exit_price is not None:
                pnl_pct = (exit_price - entry) / entry if side == "LONG" else (entry - exit_price) / entry
                gross_pnl = pos["notional"] * pnl_pct
                exit_fee = pos["notional"] * FEE_RATE
                net_pnl = gross_pnl - exit_fee - pos["entry_fee"]
                equity += gross_pnl - exit_fee

                trades.append({"symbol": symbol, "dt": row["dt"], "side": side, "pnl": net_pnl, "equity": equity})
                consecutive_losses = consecutive_losses + 1 if net_pnl < 0 else 0
                pos = None

                peak_equity = max(peak_equity, equity)
                max_dd = max(max_dd, (peak_equity - equity) / peak_equity * 100)

                if (equity - day_start_equity) / day_start_equity >= 0.12 or consecutive_losses >= 2:
                    day_locked = True

        if pos is None and not day_locked and equity > 50:
            sig = check_signal(row)
            if sig is not None:
                entry = row["close"]
                sl_dist = 2.8 * row["atr14"]
                tp_dist = 2.6 * row["atr14"]
                sl_pct = sl_dist / entry

                notional = min((equity * RISK_PER_TRADE) / sl_pct, equity * max_lev)
                entry_fee = notional * FEE_RATE
                equity -= entry_fee

                pos = {
                    "side": sig,
                    "entry": entry,
                    "sl": entry - sl_dist if sig == "LONG" else entry + sl_dist,
                    "tp": entry + tp_dist if sig == "LONG" else entry - tp_dist,
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


def run_portfolio_backtest(dfs_by_symbol):
    # 4종목 통합 1개 계좌 복리 운용 (동시 최대 2포지션, 포지션당 리스크 5.5%)
    equity = 1000.0
    peak_equity = equity
    max_dd = 0.0
    positions = {}
    trades = []

    lookups = {sym: df.set_index("dt").to_dict("index") for sym, df in dfs_by_symbol.items()}
    all_dts = sorted(set().union(*(df["dt"].tolist() for df in dfs_by_symbol.values())))

    current_day = None
    day_start_equity = equity
    day_locked = False

    for dt in all_dts:
        day = dt.date()
        if day != current_day:
            current_day = day
            day_start_equity = equity
            day_locked = False

        # 1) 보유 포지션 청산 체크
        for sym in list(positions.keys()):
            row = lookups[sym].get(dt)
            if row is None:
                continue
            pos = positions[sym]
            pos["bars"] += 1
            side = pos["side"]
            entry = pos["entry"]
            exit_price = None

            if side == "LONG":
                if row["low"] <= pos["sl"]:
                    exit_price = pos["sl"]
                elif row["high"] >= pos["tp"]:
                    exit_price = pos["tp"]
                elif pos["bars"] >= 128:
                    exit_price = row["close"]
            else:
                if row["high"] >= pos["sl"]:
                    exit_price = pos["sl"]
                elif row["low"] <= pos["tp"]:
                    exit_price = pos["tp"]
                elif pos["bars"] >= 128:
                    exit_price = row["close"]

            if exit_price is not None:
                pnl_pct = (exit_price - entry) / entry if side == "LONG" else (entry - exit_price) / entry
                gross_pnl = pos["notional"] * pnl_pct
                exit_fee = pos["notional"] * FEE_RATE
                net_pnl = gross_pnl - exit_fee - pos["entry_fee"]
                equity += gross_pnl - exit_fee

                trades.append({"symbol": sym, "dt": dt, "side": side, "pnl": net_pnl, "equity": equity})
                del positions[sym]

                peak_equity = max(peak_equity, equity)
                max_dd = max(max_dd, (peak_equity - equity) / peak_equity * 100)

                day_ret = (equity - day_start_equity) / day_start_equity
                if day_ret >= 0.12 or day_ret <= -0.10:
                    day_locked = True

        # 2) 신규 진입 체크 (동시 최대 2종목)
        if not day_locked and len(positions) < 2 and equity > 50:
            for sym, max_lev in MAX_LEVERAGE.items():
                if sym in positions or len(positions) >= 2:
                    continue
                row = lookups[sym].get(dt)
                if row is None:
                    continue
                sig = check_signal(row)
                if sig is not None:
                    entry = row["close"]
                    sl_dist = 2.8 * row["atr14"]
                    tp_dist = 2.6 * row["atr14"]
                    sl_pct = sl_dist / entry

                    notional = min((equity * 0.055) / sl_pct, equity * max_lev * 0.5)
                    entry_fee = notional * FEE_RATE
                    equity -= entry_fee

                    positions[sym] = {
                        "side": sig,
                        "entry": entry,
                        "sl": entry - sl_dist if sig == "LONG" else entry + sl_dist,
                        "tp": entry + tp_dist if sig == "LONG" else entry - tp_dist,
                        "notional": notional,
                        "entry_fee": entry_fee,
                        "bars": 0,
                    }

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        return {"trades": 0, "win_rate": 0.0, "ret_pct": 0.0, "mdd_pct": 0.0}, tdf

    win_rate = (tdf["pnl"] > 0).sum() / len(tdf) * 100
    ret_pct = (equity - 1000.0) / 1000.0 * 100
    return {
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
    raw_data = {}
    for sym in MAX_LEVERAGE:
        raw_data[sym] = fetch_15m_ohlcv(exchange, sym, days=DAYS)

    _, btc_4h_raw = build_htf_features(raw_data["BTC/USDT:USDT"])
    btc_4h = btc_4h_raw[["dt", "close_4h", "ema50_4h"]].rename(
        columns={"close_4h": "btc_close_4h", "ema50_4h": "btc_ema50_4h"}
    )

    dfs_by_sym = {}
    summaries = []
    for sym, lev in MAX_LEVERAGE.items():
        df = add_indicators(raw_data[sym].copy(), btc_4h)
        dfs_by_sym[sym] = df
        res, _ = run_single_backtest(df, sym, lev)
        summaries.append(res)

    port_res, port_tdf = run_portfolio_backtest(dfs_by_sym)
    if not port_tdf.empty:
        port_tdf.to_csv("backtest_trades.csv", index=False)

    lines = [f"📊 고빈도 눌림목+통합복리 결과 ({DAYS}일)"]
    for s in summaries:
        lines.append(
            f"- {s['symbol'].split('/')[0]}: {s['trades']}회 | 승률 {s['win_rate']}% | 누적 {s['ret_pct']}% | MDD {s['mdd_pct']}% | PF {s['pf']}"
        )
    lines.append("──────────────")
    lines.append(
        f"🔥 4종목 통합계좌: 총 {port_res['trades']}회 | 승률 {port_res['win_rate']}% | 누적 {port_res['ret_pct']}% | MDD {port_res['mdd_pct']}%"
    )
    msg = "\n".join(lines)
    print(msg)
    send_telegram(msg)


if __name__ == "__main__":
    main()
