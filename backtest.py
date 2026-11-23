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
FEE_RATE = 0.0002  # 바이비트 지정가(Maker) 편도 0.02%


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


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(df, period=14):
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def calc_adx(df, period=14):
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = calc_atr(df, period)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def prepare_15m_and_1h(raw_df, btc_1h_ref, btc_4h_ref):
    # 1시간봉 데이터 생성
    df_1h = (
        raw_df.set_index("dt")
        .resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    df_1h["ema20_1h"] = df_1h["close"].ewm(span=20, adjust=False).mean()
    df_1h["ema50_1h"] = df_1h["close"].ewm(span=50, adjust=False).mean()
    df_1h["adx14_1h"] = calc_adx(df_1h, 14)
    df_1h["rsi14_1h"] = calc_rsi(df_1h["close"], 14)
    df_1h["prev_rsi_1h"] = df_1h["rsi14_1h"].shift(1)
    df_1h["atr14_1h"] = calc_atr(df_1h, 14)
    df_1h["atr_pct_1h"] = df_1h["atr14_1h"] / df_1h["close"]
    df_1h["prev_low_1h"] = df_1h["low"].shift(1)
    df_1h["prev_high_1h"] = df_1h["high"].shift(1)

    # 4시간봉 데이터 생성
    df_4h = (
        raw_df.set_index("dt")
        .resample("4h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    df_4h["close_4h"] = df_4h["close"]
    df_4h["ema50_4h"] = df_4h["close"].ewm(span=50, adjust=False).mean()

    s_1h = df_1h[["close", "ema20_1h", "ema50_1h", "adx14_1h"]].rename(columns={"close": "close_1h"}).shift(1).reset_index()
    s_4h = df_4h[["close_4h", "ema50_4h"]].shift(1).reset_index()

    # 15분봉 프레임 완성 (모델 A용)
    df15 = raw_df.copy()
    bb_mid = df15["close"].rolling(20).mean()
    bb_std = df15["close"].rolling(20).std()
    df15["bb_upper"] = bb_mid + 2.0 * bb_std
    df15["bb_lower"] = bb_mid - 2.0 * bb_std
    df15["prev_bb_upper"] = df15["bb_upper"].shift(1)
    df15["prev_bb_lower"] = df15["bb_lower"].shift(1)
    df15["prev_high"] = df15["high"].shift(1)
    df15["prev_low"] = df15["low"].shift(1)
    df15["rsi14"] = calc_rsi(df15["close"], 14)
    df15["prev_rsi"] = df15["rsi14"].shift(1)
    df15["atr14"] = calc_atr(df15, 14)
    df15["atr_pct"] = df15["atr14"] / df15["close"]

    for ref in [s_1h, s_4h, btc_1h_ref, btc_4h_ref]:
        df15 = pd.merge_asof(df15.sort_values("dt"), ref.sort_values("dt"), on="dt", direction="backward")
    df15 = df15.dropna().reset_index(drop=True)

    # 1시간봉 프레임 완성 (모델 B용)
    df1h_exec = df_1h.reset_index()
    df1h_exec = pd.merge_asof(df1h_exec.sort_values("dt"), s_4h.sort_values("dt"), on="dt", direction="backward")
    df1h_exec = pd.merge_asof(df1h_exec.sort_values("dt"), btc_4h_ref.sort_values("dt"), on="dt", direction="backward")
    df1h_exec = df1h_exec.dropna().reset_index(drop=True)

    return df15, df1h_exec


def sig_model_a(row):
    # 모델 A: 5차 고승률 BB 외곽 반전 로직 (거래량/기울기 병목 제거)
    if row["atr_pct"] < 0.0022:
        return None
    long_trend = (
        (row["btc_close_4h"] > row["btc_ema50_4h"])
        and (row["btc_close_1h"] > row["btc_ema20_1h"])
        and (row["ema20_1h"] > row["ema50_1h"])
        and (row["adx14_1h"] >= 20)
    )
    short_trend = (
        (row["btc_close_4h"] < row["btc_ema50_4h"])
        and (row["btc_close_1h"] < row["btc_ema20_1h"])
        and (row["ema20_1h"] < row["ema50_1h"])
        and (row["adx14_1h"] >= 20)
    )

    long_touch = (row["low"] <= row["bb_lower"]) or (row["prev_low"] <= row["prev_bb_lower"])
    long_rsi = (row["prev_rsi"] <= 38) and (row["rsi14"] > row["prev_rsi"]) and (row["rsi14"] <= 48)
    if long_trend and long_touch and long_rsi and (row["close"] > row["open"]):
        return "LONG"

    short_touch = (row["high"] >= row["bb_upper"]) or (row["prev_high"] >= row["prev_bb_upper"])
    short_rsi = (row["prev_rsi"] >= 62) and (row["rsi14"] < row["prev_rsi"]) and (row["rsi14"] >= 52)
    if short_trend and short_touch and short_rsi and (row["close"] < row["open"]):
        return "SHORT"
    return None


def sig_model_b(row):
    # 모델 B: 1시간봉 실행 스윙 추세 눌림목 (노이즈/수수료 최소화)
    if row["atr_pct_1h"] < 0.004:
        return None
    long_trend = (
        (row["btc_close_4h"] > row["btc_ema50_4h"])
        and (row["close_4h"] > row["ema50_4h"])
        and (row["ema20_1h"] > row["ema50_1h"])
        and (row["adx14_1h"] >= 20)
    )
    short_trend = (
        (row["btc_close_4h"] < row["btc_ema50_4h"])
        and (row["close_4h"] < row["ema50_4h"])
        and (row["ema20_1h"] < row["ema50_1h"])
        and (row["adx14_1h"] >= 20)
    )

    long_pull = (row["low"] <= row["ema20_1h"]) or (row["prev_low_1h"] <= row["ema20_1h"])
    long_rsi = (row["prev_rsi_1h"] <= 43) and (row["rsi14_1h"] > row["prev_rsi_1h"])
    if long_trend and long_pull and long_rsi and (row["close"] > row["open"]):
        return "LONG"

    short_pull = (row["high"] >= row["ema20_1h"]) or (row["prev_high_1h"] >= row["ema20_1h"])
    short_rsi = (row["prev_rsi_1h"] >= 57) and (row["rsi14_1h"] < row["prev_rsi_1h"])
    if short_trend and short_pull and short_rsi and (row["close"] < row["open"]):
        return "SHORT"
    return None


def simulate_portfolio(dfs_by_sym, sig_fn, atr_col, sl_mult, tp_mult, max_bars, risk_per_trade):
    equity = 1000.0
    peak_equity = equity
    max_dd = 0.0
    positions = {}
    trades = []

    lookups = {sym: df.set_index("dt").to_dict("index") for sym, df in dfs_by_sym.items()}
    all_dts = sorted(set().union(*(df["dt"].tolist() for df in dfs_by_sym.values())))

    current_day = None
    day_start_equity = equity
    day_locked = False

    for dt in all_dts:
        day = dt.date()
        if day != current_day:
            current_day = day
            day_start_equity = equity
            day_locked = False

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
                elif pos["bars"] >= max_bars:
                    exit_price = row["close"]
            else:
                if row["high"] >= pos["sl"]:
                    exit_price = pos["sl"]
                elif row["low"] <= pos["tp"]:
                    exit_price = pos["tp"]
                elif pos["bars"] >= max_bars:
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
                if day_ret >= 0.12 or day_ret <= -0.08:
                    day_locked = True

        if not day_locked and len(positions) < 2 and equity > 50:
            for sym, max_lev in MAX_LEVERAGE.items():
                if sym in positions or len(positions) >= 2:
                    continue
                row = lookups[sym].get(dt)
                if row is None:
                    continue
                sig = sig_fn(row)
                if sig is not None:
                    entry = row["close"]
                    sl_dist = sl_mult * row[atr_col]
                    tp_dist = tp_mult * row[atr_col]
                    sl_pct = sl_dist / entry

                    notional = min((equity * risk_per_trade) / sl_pct, equity * max_lev * 0.5)
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
        return {"trades": 0, "win_rate": 0.0, "ret_pct": 0.0, "mdd_pct": 0.0, "pf": 0.0}, tdf

    win_rate = (tdf["pnl"] > 0).sum() / len(tdf) * 100
    ret_pct = (equity - 1000.0) / 1000.0 * 100
    gross_win = tdf.loc[tdf["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(tdf.loc[tdf["pnl"] <= 0, "pnl"].sum())
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 99.0

    return {
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
    raw_data = {sym: fetch_15m_ohlcv(exchange, sym, days=DAYS) for sym in MAX_LEVERAGE}

    btc_raw = raw_data["BTC/USDT:USDT"]
    btc_1h = btc_raw.set_index("dt").resample("1h").agg({"close": "last"}).dropna()
    btc_1h["btc_close_1h"] = btc_1h["close"]
    btc_1h["btc_ema20_1h"] = btc_1h["close"].ewm(span=20, adjust=False).mean()
    btc_1h_ref = btc_1h[["btc_close_1h", "btc_ema20_1h"]].shift(1).reset_index()

    btc_4h = btc_raw.set_index("dt").resample("4h").agg({"close": "last"}).dropna()
    btc_4h["btc_close_4h"] = btc_4h["close"]
    btc_4h["btc_ema50_4h"] = btc_4h["close"].ewm(span=50, adjust=False).mean()
    btc_4h_ref = btc_4h[["btc_close_4h", "btc_ema50_4h"]].shift(1).reset_index()

    dfs_a, dfs_b = {}, {}
    for sym in MAX_LEVERAGE:
        d15, d1h = prepare_15m_and_1h(raw_data[sym], btc_1h_ref, btc_4h_ref)
        dfs_a[sym] = d15
        dfs_b[sym] = d1h

    res_a, tdf_a = simulate_portfolio(dfs_a, sig_model_a, "atr14", sl_mult=2.8, tp_mult=2.6, max_bars=128, risk_per_trade=0.035)
    res_b, tdf_b = simulate_portfolio(dfs_b, sig_model_b, "atr14_1h", sl_mult=2.5, tp_mult=2.5, max_bars=48, risk_per_trade=0.04)

    if not tdf_a.empty or not tdf_b.empty:
        tdf_a["model"] = "A_15m_BB"
        tdf_b["model"] = "B_1h_Swing"
        pd.concat([tdf_a, tdf_b], ignore_index=True).to_csv("backtest_trades.csv", index=False)

    msg = "\n".join([
        f"📊 4종목 통합계좌 비교 결과 ({DAYS}일)",
        f"▶ 모델A (15m BB반전): {res_a['trades']}회 | 승률 {res_a['win_rate']}% | 누적 {res_a['ret_pct']}% | MDD {res_a['mdd_pct']}% | PF {res_a['pf']}",
        f"▶ 모델B (1H 스윙눌림): {res_b['trades']}회 | 승률 {res_b['win_rate']}% | 누적 {res_b['ret_pct']}% | MDD {res_b['mdd_pct']}% | PF {res_b['pf']}",
    ])
    print(msg)
    send_telegram(msg)


if __name__ == "__main__":
    main()
