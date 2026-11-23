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


def build_features(raw_data, tf="15m"):
    # BTC 기준 상위 추세 생성
    btc_raw = raw_data["BTC/USDT:USDT"]
    btc_4h = btc_raw.set_index("dt").resample("4h").agg({"close": "last"}).dropna()
    btc_4h["btc_c4"] = btc_4h["close"]
    btc_4h["btc_e50_4"] = btc_4h["close"].ewm(span=50, adjust=False).mean()
    btc_4h_ref = btc_4h[["btc_c4", "btc_e50_4"]].shift(1).reset_index()

    btc_1h = btc_raw.set_index("dt").resample("1h").agg({"close": "last"}).dropna()
    btc_1h["btc_c1"] = btc_1h["close"]
    btc_1h["btc_e20_1"] = btc_1h["close"].ewm(span=20, adjust=False).mean()
    btc_1h_ref = btc_1h[["btc_c1", "btc_e20_1"]].shift(1).reset_index()

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
        df_4h["e20_4"] = df_4h["close"].ewm(span=20, adjust=False).mean()
        df_4h["e50_4"] = df_4h["close"].ewm(span=50, adjust=False).mean()

        if tf == "15m":
            df = raw.copy()
            s_1h = df_1h[["e20_1", "e50_1", "adx_1"]].shift(1).reset_index()
            s_4h = df_4h[["c4", "e20_4", "e50_4"]].shift(1).reset_index()
            for ref in [s_1h, s_4h, btc_1h_ref, btc_4h_ref]:
                df = pd.merge_asof(df.sort_values("dt"), ref.sort_values("dt"), on="dt", direction="backward")
        else:
            df = df_1h.reset_index()
            s_4h = df_4h[["c4", "e20_4", "e50_4"]].shift(1).reset_index()
            for ref in [s_4h, btc_1h_ref, btc_4h_ref]:
                df = pd.merge_asof(df.sort_values("dt"), ref.sort_values("dt"), on="dt", direction="backward")

        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        bb_mid = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_u"] = bb_mid + 2.0 * bb_std
        df["bb_l"] = bb_mid - 2.0 * bb_std
        df["rsi"] = calc_rsi(df["close"], 14)
        df["atr"] = calc_atr(df, 14)
        df["atr_pct"] = df["atr"] / df["close"]
        df["vol_ma"] = df["volume"].rolling(20).mean()

        # 스토캐스틱 RSI (빠른 반전 포착용)
        rsi_min = df["rsi"].rolling(14).min()
        rsi_max = df["rsi"].rolling(14).max()
        df["stoch_k"] = ((df["rsi"] - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100).rolling(3).mean()

        dfs[sym] = df.dropna().reset_index(drop=True)

    # 공통 타임라인 정렬
    common_dts = sorted(set.intersection(*(set(dfs[s]["dt"]) for s in SYMBOLS)))
    aligned = {s: dfs[s][dfs[s]["dt"].isin(common_dts)].sort_values("dt").reset_index(drop=True) for s in SYMBOLS}
    return aligned, pd.Series(common_dts)


def generate_signal_sets(aligned, tf):
    n = len(next(iter(aligned.values())))
    presets = []

    trend_modes = ["BTC4H+Coin1H", "Strict_4H+1H", "Coin_Only"]
    strat_modes = ["BB_Reversal", "EMA_StochRSI", "Keltner_Pullback"]

    for t_mode in trend_modes:
        for s_mode in strat_modes:
            sig_mat = np.zeros((n, 4), dtype=np.int8)
            for c_idx, sym in enumerate(SYMBOLS):
                df = aligned[sym]
                min_atr = 0.0020 if tf == "15m" else 0.0038
                vol_ok = df["atr_pct"] >= min_atr

                if t_mode == "BTC4H+Coin1H":
                    up_tr = (df["btc_c4"] > df["btc_e50_4"]) & (df["e20_1"] > df["e50_1"]) & (df["adx_1"] >= 18)
                    dn_tr = (df["btc_c4"] < df["btc_e50_4"]) & (df["e20_1"] < df["e50_1"]) & (df["adx_1"] >= 18)
                elif t_mode == "Strict_4H+1H":
                    up_tr = (df["btc_c4"] > df["btc_e50_4"]) & (df["c4"] > df["e50_4"]) & (df["e20_1"] > df["e50_1"]) & (df["adx_1"] >= 20)
                    dn_tr = (df["btc_c4"] < df["btc_e50_4"]) & (df["c4"] < df["e50_4"]) & (df["e20_1"] < df["e50_1"]) & (df["adx_1"] >= 20)
                else:
                    up_tr = (df["c4"] > df["e50_4"]) & (df["e20_1"] > df["e50_1"]) & (df["adx_1"] >= 20)
                    dn_tr = (df["c4"] < df["e50_4"]) & (df["e20_1"] < df["e50_1"]) & (df["adx_1"] >= 20)

                if s_mode == "BB_Reversal":
                    touch_l = (df["low"] <= df["bb_l"]) | (df["low"].shift(1) <= df["bb_l"].shift(1))
                    rsi_l = (df["rsi"].shift(1) <= 38) & (df["rsi"] > df["rsi"].shift(1))
                    long_c = vol_ok & up_tr & touch_l & rsi_l & (df["close"] > df["open"])

                    touch_s = (df["high"] >= df["bb_u"]) | (df["high"].shift(1) >= df["bb_u"].shift(1))
                    rsi_s = (df["rsi"].shift(1) >= 62) & (df["rsi"] < df["rsi"].shift(1))
                    short_c = vol_ok & dn_tr & touch_s & rsi_s & (df["close"] < df["open"])

                elif s_mode == "EMA_StochRSI":
                    pull_l = (df["low"] <= df["ema20"]) & (df["close"] > df["ema50"])
                    st_l = (df["stoch_k"].shift(1) <= 25) & (df["stoch_k"] > df["stoch_k"].shift(1))
                    long_c = vol_ok & up_tr & pull_l & st_l & (df["close"] > df["open"])

                    pull_s = (df["high"] >= df["ema20"]) & (df["close"] < df["ema50"])
                    st_s = (df["stoch_k"].shift(1) >= 75) & (df["stoch_k"] < df["stoch_k"].shift(1))
                    short_c = vol_ok & dn_tr & pull_s & st_s & (df["close"] < df["open"])

                else:  # Keltner_Pullback
                    kc_l = df["ema20"] - 1.5 * df["atr"]
                    kc_u = df["ema20"] + 1.5 * df["atr"]
                    long_c = vol_ok & up_tr & (df["low"] <= kc_l) & (df["close"] > df["open"]) & (df["rsi"] < 48)
                    short_c = vol_ok & dn_tr & (df["high"] >= kc_u) & (df["close"] < df["open"]) & (df["rsi"] > 52)

                l_arr = long_c.fillna(False).to_numpy()
                s_arr = short_c.fillna(False).to_numpy()
                sig_mat[l_arr, c_idx] = 1
                sig_mat[s_arr & (~l_arr), c_idx] = -1

            presets.append((f"{tf}|{s_mode}|{t_mode}", sig_mat))
    return presets


def fast_simulate(high, low, close, atr, day_arr, sig_mat, tp_mult, sl_mult, max_bars, risk_pct=0.055, record_trades=False):
    n = len(close)
    equity = 1000.0
    peak = 1000.0
    max_dd = 0.0

    # 포지션 상태: active(bool), side(1/-1), entry, sl, tp, notional, entry_fee, bars
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

        # 1) 보유 포지션 청산 체크
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
                elif p_bars[c] >= max_bars:
                    exit_p = close[t, c]
            else:
                if high[t, c] >= p_sl[c]:
                    exit_p = p_sl[c]
                elif low[t, c] <= p_tp[c]:
                    exit_p = p_tp[c]
                elif p_bars[c] >= max_bars:
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
                    trade_logs.append({"bar": t, "symbol": SYMBOLS[c], "side": "LONG" if s == 1 else "SHORT", "pnl": round(net, 2), "equity": round(equity, 2)})

                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak * 100.0
                if dd > max_dd:
                    max_dd = dd

                d_ret = (equity - day_start) / day_start
                if d_ret >= 0.12 or d_ret <= -0.09:
                    day_locked = True

        # 2) 신규 진입 (동시 최대 2종목)
        if not day_locked and equity > 50.0 and active.sum() < 2:
            for c in range(4):
                if active[c] or active.sum() >= 2:
                    continue
                sig = sig_mat[t, c]
                if sig != 0:
                    ent = close[t, c]
                    sl_dist = sl_mult * atr[t, c]
                    tp_dist = tp_mult * atr[t, c]
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

    tp_sl_grid = [
        (1.6, 2.2),
        (1.8, 2.2),
        (2.0, 2.4),
        (2.2, 2.0),
        (2.5, 2.5),
        (3.0, 2.5),
    ]

    all_results = []
    cache = {}

    for tf in ["15m", "1h"]:
        aligned, dts = build_features(raw_data, tf=tf)
        high = np.column_stack([aligned[s]["high"].to_numpy() for s in SYMBOLS])
        low = np.column_stack([aligned[s]["low"].to_numpy() for s in SYMBOLS])
        close = np.column_stack([aligned[s]["close"].to_numpy() for s in SYMBOLS])
        atr = np.column_stack([aligned[s]["atr"].to_numpy() for s in SYMBOLS])
        day_arr = (dts.dt.dayofyear + dts.dt.year * 1000).to_numpy()
        max_bars = 128 if tf == "15m" else 48

        presets = generate_signal_sets(aligned, tf)
        for name, sig_mat in presets:
            for tp_m, sl_m in tp_sl_grid:
                res = fast_simulate(high, low, close, atr, day_arr, sig_mat, tp_m, sl_m, max_bars)
                res["name"] = f"{name} (TP {tp_m}/SL {sl_m})"
                all_results.append(res)
                cache[res["name"]] = (high, low, close, atr, day_arr, sig_mat, tp_m, sl_m, max_bars)

    res_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ["coin_wins", "coin_losses", "coin_pnl", "logs"]} for r in all_results])
    res_df = res_df.sort_values(by=["ret_pct", "win_rate"], ascending=False)
    res_df.to_csv("optimization_results.csv", index=False)

    # 승률 60% 이상 & 거래 30회 이상 필터링
    qualified = [r for r in all_results if r["win_rate"] >= 60.0 and r["trades"] >= 30]
    qualified.sort(key=lambda x: x["ret_pct"], reverse=True)

    if not qualified:
        qualified = [r for r in all_results if r["win_rate"] >= 58.0 and r["trades"] >= 20]
        qualified.sort(key=lambda x: x["ret_pct"], reverse=True)

    if not qualified:
        qualified = sorted(all_results, key=lambda x: x["ret_pct"], reverse=True)

    top3 = qualified[:3]
    best = top3[0]

    # 1위 전략 상세 로그 저장
    h, l, c, a, d_arr, s_mat, tp_m, sl_m, m_bars = cache[best["name"]]
    best_detail = fast_simulate(h, l, c, a, d_arr, s_mat, tp_m, sl_m, m_bars, record_trades=True)
    pd.DataFrame(best_detail["logs"]).to_csv("backtest_trades.csv", index=False)

    lines = [f"🏆 108개 전략 전수탐색 TOP 3 ({DAYS}일 통합계좌)"]
    for idx, r in enumerate(top3, 1):
        lines.append(
            f"{idx}위: {r['name']}\n"
            f"   ▶ {r['trades']}회 | 승률 {r['win_rate']}% | 누적 {r['ret_pct']}% | MDD {r['mdd_pct']}% | PF {r['pf']}"
        )

    lines.append("──────────────")
    lines.append("📌 1위 전략 종목별 성과:")
    for i, sym in enumerate(SYMBOLS):
        w = best_detail["coin_wins"][i]
        lo = best_detail["coin_losses"][i]
        tot = w + lo
        wr = round(w / tot * 100, 1) if tot > 0 else 0.0
        pnl = round(best_detail["coin_pnl"][i], 1)
        lines.append(f"- {sym.split('/')[0]}: {tot}회 (승률 {wr}%, 순손익 {pnl}$)")

    msg = "\n".join(lines)
    print(msg)
    send_telegram(msg)


if __name__ == "__main__":
    main()
