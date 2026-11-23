import os
import json
import time
from datetime import datetime, timezone
import requests
import ccxt
import numpy as np
import pandas as pd

SYMBOLS = {
    "BTC/USDT:USDT": {"strategy": "KELTNER", "max_lev": 15.0, "tp_mult": 2.2, "sl_mult": 2.0},
    "ETH/USDT:USDT": {"strategy": "KELTNER", "max_lev": 15.0, "tp_mult": 2.2, "sl_mult": 2.0},
    "SOL/USDT:USDT": {"strategy": "EMA_STOCH", "max_lev": 10.0, "tp_mult": 2.5, "sl_mult": 2.5},
    "ADA/USDT:USDT": {"strategy": "EMA_STOCH", "max_lev": 10.0, "tp_mult": 2.5, "sl_mult": 2.5},
}

RISK_PER_TRADE = 0.055  # 1회 손절 시 계좌 리스크 5.5%
LAST_ALERT_FILE = "last_alert.json"
HEARTBEAT_FILE = "heartbeat.txt"


def load_last_alerts():
    if os.path.exists(LAST_ALERT_FILE):
        try:
            with open(LAST_ALERT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_last_alerts(data):
    with open(LAST_ALERT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_heartbeat():
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(HEARTBEAT_FILE, "w", encoding="utf-8") as f:
        f.write(now_str + "\n")


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


def format_price(p):
    if p >= 1000:
        return f"{p:,.1f}"
    elif p >= 10:
        return f"{p:,.2f}"
    return f"{p:,.4f}"


def fetch_and_prepare(exchange, symbol):
    rows = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=300)
    df_1h = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df_1h = df_1h.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    df_1h["dt"] = pd.to_datetime(df_1h["ts"], unit="ms", utc=True)

    df_1h["e20_1"] = df_1h["close"].ewm(span=20, adjust=False).mean()
    df_1h["e50_1"] = df_1h["close"].ewm(span=50, adjust=False).mean()
    df_1h["adx_1"] = calc_adx(df_1h, 14)

    df_4h = (
        df_1h.set_index("dt")
        .resample("4h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    df_4h["c4"] = df_4h["close"]
    df_4h["e50_4"] = df_4h["close"].ewm(span=50, adjust=False).mean()
    s_4h = df_4h[["c4", "e50_4"]].shift(1).reset_index()

    df = pd.merge_asof(df_1h.sort_values("dt"), s_4h.sort_values("dt"), on="dt", direction="backward")
    df["rsi"] = calc_rsi(df["close"], 14)
    df["atr"] = calc_atr(df, 14)
    df["atr_pct"] = df["atr"] / df["close"]

    rsi_min = df["rsi"].rolling(14).min()
    rsi_max = df["rsi"].rolling(14).max()
    df["stoch_k"] = ((df["rsi"] - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100).rolling(3).mean()

    return df.dropna().reset_index(drop=True)


def evaluate_signal(df, cfg, coin):
    if len(df) < 5:
        return None, f"- {coin}: 데이터 부족"

    row = df.iloc[-2]  # 직전 마감 1시간봉 기준
    prev = df.iloc[-3]
    price_str = format_price(float(row["close"]))
    adx_val = float(row["adx_1"])
    candle_dt = row["dt"].strftime("%Y-%m-%d %H:%M UTC")

    up_tr = (row["c4"] > row["e50_4"]) and (row["e20_1"] > row["e50_1"]) and (adx_val >= 20)
    dn_tr = (row["c4"] < row["e50_4"]) and (row["e20_1"] < row["e50_1"]) and (adx_val >= 20)

    if up_tr:
        trend_str = f"상승추세(ADX {adx_val:.1f})"
    elif dn_tr:
        trend_str = f"하락추세(ADX {adx_val:.1f})"
    else:
        trend_str = f"횡보/혼조(ADX {adx_val:.1f})"

    side = None
    if cfg["strategy"] == "KELTNER":
        kc_l = float(row["e20_1"] - 1.5 * row["atr"])
        kc_u = float(row["e20_1"] + 1.5 * row["atr"])
        rsi_val = float(row["rsi"])
        ind_str = f"RSI {rsi_val:.1f} | KC {format_price(kc_l)}~{format_price(kc_u)}"

        if row["atr_pct"] >= 0.0038:
            if up_tr and (row["low"] <= kc_l) and (row["close"] > row["open"]) and (rsi_val < 48):
                side = "LONG"
            elif dn_tr and (row["high"] >= kc_u) and (row["close"] < row["open"]) and (rsi_val > 52):
                side = "SHORT"
    else:
        e20_val = float(row["e20_1"])
        st_val = float(row["stoch_k"])
        ind_str = f"StochK {st_val:.1f} | EMA20 {format_price(e20_val)}"

        if row["atr_pct"] >= 0.0038:
            pull_l = (row["low"] <= e20_val) and (row["close"] > row["e50_1"])
            st_l = (prev["stoch_k"] <= 25) and (st_val > prev["stoch_k"])
            if up_tr and pull_l and st_l and (row["close"] > row["open"]):
                side = "LONG"
            else:
                pull_s = (row["high"] >= e20_val) and (row["close"] < row["e50_1"])
                st_s = (prev["stoch_k"] >= 75) and (st_val < prev["stoch_k"])
                if dn_tr and pull_s and st_s and (row["close"] < row["open"]):
                    side = "SHORT"

    state_tag = f"🚨 {side} 진입" if side else "⏳ 대기"
    status_line = f"- {coin}: {price_str} | {trend_str} | {ind_str} → {state_tag}"

    if side is None:
        return None, status_line

    entry = float(row["close"])
    atr = float(row["atr"])
    sl_dist = cfg["sl_mult"] * atr
    tp_dist = cfg["tp_mult"] * atr

    sl_price = entry - sl_dist if side == "LONG" else entry + sl_dist
    tp_price = entry + tp_dist if side == "LONG" else entry - tp_dist

    sl_pct = sl_dist / entry
    tp_pct = tp_dist / entry
    max_lev = cfg["max_lev"]

    notional_mult = min(RISK_PER_TRADE / sl_pct, max_lev * 0.5)
    margin_pct = (notional_mult / max_lev) * 100.0

    return {
        "candle_ts": int(row["ts"]),
        "candle_dt": candle_dt,
        "side": side,
        "entry": entry,
        "tp": tp_price,
        "sl": sl_price,
        "tp_pct": tp_pct * 100.0,
        "sl_pct": sl_pct * 100.0,
        "max_lev": int(max_lev),
        "margin_pct": round(margin_pct, 1),
        "strategy": cfg["strategy"],
    }, status_line


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
    last_alerts = load_last_alerts()
    updated = False
    is_manual = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"

    status_lines = []
    latest_dt = ""

    for sym, cfg in SYMBOLS.items():
        coin = sym.split("/")[0]
        try:
            df = fetch_and_prepare(exchange, sym)
            if len(df) >= 2:
                latest_dt = df.iloc[-2]["dt"].strftime("%Y-%m-%d %H:%M UTC")
            sig, status_line = evaluate_signal(df, cfg, coin)
            status_lines.append(status_line)
            time.sleep(0.2)

            if sig is None:
                continue

            if last_alerts.get(sym) == sig["candle_ts"]:
                continue

            emoji = "🟢" if sig["side"] == "LONG" else "🔴"
            msg = (
                f"{emoji} [{coin}USDT] {sig['side']} 진입 시그널\n"
                f"• 전략: {sig['strategy']} (1H 마감기준)\n"
                f"• 기준봉: {sig['candle_dt']}\n"
                f"• 진입가: {format_price(sig['entry'])}\n"
                f"• 익절가(TP): {format_price(sig['tp'])} (+{sig['tp_pct']:.2f}%)\n"
                f"• 손절가(SL): {format_price(sig['sl'])} (-{sig['sl_pct']:.2f}%)\n"
                f"• 바이비트 설정: 격리 {sig['max_lev']}배 | 총 시드의 {sig['margin_pct']}% 증거금 투입"
            )
            print(msg)
            send_telegram(msg)

            last_alerts[sym] = sig["candle_ts"]
            updated = True
        except Exception as e:
            err_line = f"- {coin}: 조회 오류 ({e})"
            print(err_line)
            status_lines.append(err_line)

    if is_manual:
        report = "\n".join([
            f"🔍 [수동 실행 상태 점검] ({latest_dt} 마감봉)",
            *status_lines,
        ])
        print(report)
        send_telegram(report)

    if updated:
        save_last_alerts(last_alerts)
    update_heartbeat()


if __name__ == "__main__":
    main()
