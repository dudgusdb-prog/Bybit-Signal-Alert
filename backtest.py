import os
import time
import requests
import ccxt
import numpy as np
import pandas as pd

SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "ADA/USDT:USDT"]
MAX_LEV = np.array([15.0, 15.0, 10.0, 10.0])
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


def build_1h_features(raw_data):
    dfs = {}
    for sym in SYMBOLS:
        raw = raw_data[sym]
        df_1h = (
            raw.set_index("dt")
            .resample("1h")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
        df_1h["e20_1"] = df_1h["close"].ewm(span=20, adjust=False).mean()
        df_1h["e50_1"] = df_1h["close"].ewm(span=50, adjust=False).mean()
        df_1h["adx_1"] = calc_adx(df_1h, 14)

        df_4h = (
            raw.set_index("dt")
            .resample("4h")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
        )
        df_4h["c4"] = df_4h["close"]
        df_4h["e50_4"] = df_4h["close"].ewm(span=50, adjust=False).mean()

        df = df_1h.reset_index()
        s_4h = df_4h[["c4", "e50_4"]].shift(1).reset_index()
        df = pd.merge_asof(df.sort_values("dt"), s_4h.sort_values("dt"), on="dt", direction="backward")

        df["ema20"] = df["e20_1"]
        df["ema50"] = df["e50_1"]
        df["rsi"] = calc_rsi(df["close"], 14)
        df["atr"] = calc_atr(df, 14)
        df["atr_pct"] = df["atr"] / df["close"]

        rsi_min = df["rsi"].rolling(14).min()
        rsi_max = df["rsi"].rolling(14).max()
        df["stoch_k"] = ((df["rsi"] - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100).rolling(3).mean()

        dfs[sym] = df.dropna().reset_index(drop=True)

    common_dts = sorted(set.intersection(*(set(dfs[s]["dt"]) for s in SYMBOLS)))
    aligned = {s: dfs[s][dfs[s]["dt"].isin(common_dts)].sort_values("dt").reset_index(drop=True) for s in SYMBOLS}
    return aligned, pd.Series(common_dts)


def build_combo_matrices(aligned):
    n = len(next(iter(aligned.values())))
    sig1 = np.zeros((n, 4), dtype=np.int8)  # 1위: 1h EMA_StochRSI
    sig2 = np.zeros((n, 4), dtype=np.int8)  # 2위: 1h Keltner_Pullback

    for c_idx, sym in enumerate(SYMBOLS):
        df = aligned[sym]
        vol_ok = df["atr_pct"] >= 0.0038
        up_tr = (df["c4"] > df["e50_4"]) & (df["e20_1"] > df["e50_1"]) & (df["adx_1"] >= 20)
        dn_tr = (df["c4"] < df["e50_4"]) & (df["e20_1"] < df["e50_1"]) & (df["adx_1"] >= 20)

        # 1위 전략 신호
        pull_l = (df["low"] <= df["ema20"]) & (df["close"] > df["ema50"])
        st_l = (df["stoch_k"].shift(1) <= 25) & (df["stoch_k"] > df["stoch_k"].shift(1))
        l1 = (vol_ok & up_tr & pull_l & st_l & (df["close"] > df["open"])).fillna(False).to_numpy()

        pull_s = (df["high"] >= df["ema20"]) & (df["close"] < df["ema50"])
        st_s = (df["stoch_k"].shift(1) >= 75) & (df["stoch_k"] < df["stoch_k"].shift(1))
        s1 = (vol_ok & dn_tr & pull_s & st_s & (df["close"] < df["open"])).fillna(False).to_numpy()

        sig1[l1, c_idx] = 1
        sig1[s1 & (~l1), c_idx] = -1

        # 2위 전략 신호
        kc_l = df["ema20"] - 1.5 * df["atr"]
        kc_u = df["ema20"] + 1.5 * df["atr"]
        l2 = (vol_ok & up_tr & (df["low"] <= kc_l) & (df["close"] > df["open"]) & (df["rsi"] < 48)).fillna(False).to_numpy()
        s2 = (vol_ok & dn_tr & (df["high"] >= kc_u) & (df["close"] < df["open"]) & (df["rsi"] > 52)).fillna(False).to_numpy()

        sig2[l2, c_idx] = 1
        sig2[s2 & (~l2), c_idx] = -1

    # 조합 ①: 종목별 맞춤 배분 (BTC·ETH -> #2 [2.2/2.0], SOL·ADA -> #1 [2.5/2.5])
    c1_sig = np.zeros((n, 4), dtype=np.int8)
    c1_tp = np.zeros((n, 4))
    c1_sl = np.zeros((n, 4))
    c1_sig[:, :2] = sig2[:, :2]
    c1_tp[:, :2] = 2.2
    c1_sl[:, :2] = 2.0
    c1_sig[:, 2:] = sig1[:, 2:]
    c1_tp[:, 2:] = 2.5
    c1_sl[:, 2:] = 2.5

    # 조합 ②: OR 병행 (#2 우선, 미발생 시 #1 적용) + 리스크 3.5%
    c2_sig = np.where(sig2 != 0, sig2, sig1)
    c2_tp = np.where(sig2 != 0, 2.2, 2.5)
    c2_sl = np.where(sig2 != 0, 2.0, 2.5)

    # 조합 ③: AND 교집합 (두 신호 동시 일치 시 진입) + 리스크 5.5%
    c3_sig = np.where((sig1 != 0) & (sig1 == sig2), sig1, 0)
    c3_tp = np.full((n, 4), 2.35)
    c3_sl = np.full((n, 4), 2.25)

    return [
        ("조합① 종목맞춤 (BTC·ETH=#2 / SOL·ADA=#1)", c1_sig, c1_tp, c1_sl, 0.055),
        ("조합② OR병행 + 리스크 3.5% 축소", c2_sig, c2_tp, c2_sl, 0.035),
        ("조합③ AND교집합 (동시일치 진입)", c3_sig, c3_tp, c3_sl, 0.055),
    ]


def simulate_combo(high, low, close, atr, day_arr, sig_mat, tp_mat, sl_mat, risk_pct, record_trades=False):
    n = len(close)
    equity = 1000.0
    peak = 1000.0
    max_dd = 0.0

    active = np.zeros(4, dtype=bool)
    p_side = np.zeros(4, dtype=np.int8)
    p_entry = np.zeros(4)
    p_sl = np.zeros(4)
    p_tp = np.zeros(4)
    p_notional = np.zeros(4)
    p_efee = np.zeros(4)
    p_bars = np.zeros(4, dtype=np.int32)

    cur_day = -1
    day_start = 1000.0
    day_locked = False

    wins = 0
    losses = 0
    gross_win = 0.0
    gross_loss = 0.0
    coin_wins = np.zeros(4, dtype=np.int32)
    coin_losses = np.zeros(4, dtype=np.int32)
    coin_pnl = np.zeros(4)
    trade_logs = []

    for t in range(n):
        d = day_arr[t]
        if d != cur_day:
            cur_day = d
            day_start = equity
            day_locked = False

        for c in range(4):
            if not active[c]:
                continue
            p_bars[c] += 1
            s = p_side[c]
            exit_p = 0.0
            if s == 1:
                if low[t, c] <= p_sl[c]:
                    exit_p = p_sl[c]
                elif high[t, c] >= p_tp[c]:
                    exit_p = p_tp[c]
                elif p_bars[c] >= 48:
                    exit_p = close[t, c]
            else:
                if high[t, c] >= p_sl[c]:
                    exit_p = p_sl[c]
                elif low[t, c] <= p_tp[c]:
                    exit_p = p_tp[c]
                elif p_bars[c] >= 48:
                    exit_p = close[t, c]

            if exit_p > 0.0:
                pnl_pct = (exit_p - p_entry[c]) / p_entry[c] if s == 1 else (p_entry[c] - exit_p) / p_entry[c]
                gpnl = p_notional[c] * pnl_pct
                xfee = p_notional[c] * FEE_RATE
                net = gpnl - xfee - p_efee[c]
                equity += gpnl - xfee
                active[c] = False

                coin_pnl[c] += net
                if net > 0:
                    wins += 1
                    coin_wins[c] += 1
                    gross_win += net
                else:
                    losses += 1
                    coin_losses[c] += 1
                    gross_loss += abs(net)

                if record_trades:
                    trade_logs.append({
                        "bar": t,
                        "symbol": SYMBOLS[c],
                        "side": "LONG" if s == 1 else "SHORT",
                        "pnl": round(net, 2),
                        "equity": round(equity, 2),
                    })

                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak * 100.0
                if dd > max_dd:
                    max_dd = dd

                d_ret = (equity - day_start) / day_start
                if d_ret >= 0.12 or d_ret <= -0.09:
                    day_locked = True

        if not day_locked and equity > 50.0 and active.sum() < 2:
            for c in range(4):
                if active[c] or active.sum() >= 2:
                    continue
                sig = sig_mat[t, c]
                if sig != 0:
                    ent = close[t, c]
                    sl_dist = sl_mat[t, c] * atr[t, c]
                    tp_dist = tp_mat[t, c] * atr[t, c]
                    sl_pct = sl_dist / ent
                    if sl_pct <= 1e-6:
                        continue
                    notional = min((equity * risk_pct) / sl_pct, equity * MAX_LEV[c] * 0.5)
                    efee = notional * FEE_RATE
                    equity -= efee

                    active[c] = True
                    p_side[c] = sig
                    p_entry[c] = ent
                    p_sl[c] = ent - sl_dist if sig == 1 else ent + sl_dist
                    p_tp[c] = ent + tp_dist if sig == 1 else ent - tp_dist
                    p_notional[c] = notional
                    p_efee[c] = efee
                    p_bars[c] = 0

    total_trades = wins + losses
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
    ret_pct = (equity - 1000.0) / 1000.0 * 100.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else 99.0

    return {
        "trades": total_trades,
        "win_rate": round(win_rate, 2),
        "ret_pct": round(ret_pct, 2),
        "mdd_pct": round(max_dd, 2),
        "pf": round(pf, 2),
        "coin_wins": coin_wins,
        "coin_losses": coin_losses,
        "coin_pnl": coin_pnl,
        "logs": trade_logs,
    }


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
    raw_data = {sym: fetch_15m_ohlcv(exchange, sym, days=DAYS) for sym in SYMBOLS}

    aligned, dts = build_1h_features(raw_data)
    high = np.column_stack([aligned[s]["high"].to_numpy() for s in SYMBOLS])
    low = np.column_stack([aligned[s]["low"].to_numpy() for s in SYMBOLS])
    close = np.column_stack([aligned[s]["close"].to_numpy() for s in SYMBOLS])
    atr = np.column_stack([aligned[s]["atr"].to_numpy() for s in SYMBOLS])
    day_arr = (dts.dt.dayofyear + dts.dt.year * 1000).to_numpy()

    combos = build_combo_matrices(aligned)
    results = []

    for label, sig_m, tp_m, sl_m, r_pct in combos:
        res = simulate_combo(high, low, close, atr, day_arr, sig_m, tp_m, sl_m, r_pct, record_trades=True)
        res["label"] = label
        results.append(res)

    best = max(results, key=lambda x: x["ret_pct"])
    if best["logs"]:
        pd.DataFrame(best["logs"]).to_csv("backtest_trades.csv", index=False)

    lines = [f"🔀 1·2위 전략 3가지 조합 비교 ({DAYS}일 통합계좌)"]
    for r in results:
        lines.append(
            f"▶ {r['label']}\n"
            f"   {r['trades']}회 | 승률 {r['win_rate']}% | 누적 {r['ret_pct']}% | MDD {r['mdd_pct']}% | PF {r['pf']}"
        )

    lines.append("──────────────")
    lines.append(f"📌 최고 수익({best['label'].split()[0]}) 종목별 성과:")
    for i, sym in enumerate(SYMBOLS):
        w = best["coin_wins"][i]
        lo = best["coin_losses"][i]
        tot = w + lo
        wr = round(w / tot * 100, 1) if tot > 0 else 0.0
        pnl = round(best["coin_pnl"][i], 1)
        lines.append(f"- {sym.split('/')[0]}: {tot}회 (승률 {wr}%, 순손익 {pnl}$)")

    msg = "\n".join(lines)
    print(msg)
    send_telegram(msg)


if __name__ == "__main__":
    main()
