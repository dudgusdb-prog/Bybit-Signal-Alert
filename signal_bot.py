import os
import json
import numpy as np
import pandas as pd
import requests

TG_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = str(os.environ["TELEGRAM_CHAT_ID"])
STATE_FILE = "last_alert.json"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"]

def send_tg(text: str):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    resp = requests.post(url, json={"chat_id": CHAT_ID, "text": text}, timeout=10)
    resp.raise_for_status()

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            st = json.load(f)
            st.setdefault("last_ts", {})
            st.setdefault("active_pos", None)
            st.setdefault("consec_losses", 0)
            return st
    return {"last_ts": {}, "active_pos": None, "consec_losses": 0}

def save_state(st: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=2)

def fetch_klines(symbol: str, interval: str, limit: int = 80) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    inst_id = f"{symbol.replace('USDT', '')}-USDT-SWAP"
    bar = "1H" if interval == "60" else f"{interval}m"
    try:
        r = requests.get(
            "https://www.okx.com/api/v5/market/candles",
            params={"instId": inst_id, "bar": bar, "limit": limit},
            headers=headers,
            timeout=10
        )
        data = r.json()
        if r.status_code == 200 and data.get("code") == "0":
            rows = data["data"]
            df = pd.DataFrame(rows).iloc[:, :6]
            df.columns = ["ts", "open", "high", "low", "close", "volume"]
            df = df.iloc[::-1].reset_index(drop=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            return df
    except Exception:
        pass

    product_id = f"{symbol.replace('USDT', '')}-USD"
    granularity = 3600 if interval == "60" else int(interval) * 60
    r = requests.get(
        f"https://api.exchange.coinbase.com/products/{product_id}/candles",
        params={"granularity": granularity},
        headers=headers,
        timeout=10
    )
    r.raise_for_status()
    rows = r.json()[:limit]
    df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    df = df.iloc[::-1].reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df

def format_price(symbol: str, price: float) -> str:
    if symbol == "ADAUSDT":
        return f"${price:,.4f}"
    return f"${price:,.2f}"

def track_active_position(st: dict) -> bool:
    """보유 중인 가상 포지션이 있으면 TP/SL 터치 여부를 확인하고 종료 시 알림 발송"""
    pos = st.get("active_pos")
    if not pos:
        return False

    sym = pos["symbol"]
    df_5m = fetch_klines(sym, "5", 30)
    entry_ts = int(pos["ts"])
    recent = df_5m[df_5m["ts"].astype(np.int64) > entry_ts]
    if recent.empty:
        return True

    max_high = float(recent["high"].max())
    min_low = float(recent["low"].min())
    is_long = pos["side"].startswith("LONG")

    hit_tp = (max_high >= pos["tp"]) if is_long else (min_low <= pos["tp"])
    hit_sl = (min_low <= pos["sl"]) if is_long else (max_high >= pos["sl"])

    if hit_sl or hit_tp:
        if hit_sl:
            st["consec_losses"] += 1
            next_weight = "10% (2연패 방어 모드)" if st["consec_losses"] >= 2 else "20% (기본 비중)"
            send_tg(
                f"🛑 [{sym}] 기준 손절(SL) 도달!\n"
                f"• 방향: {pos['side']}\n"
                f"• 진입가: {format_price(sym, pos['entry'])} → 손절가: {format_price(sym, pos['sl'])} (-0.95%)\n"
                f"• 현재 누적 연패: {st['consec_losses']}연패\n"
                f"• 다음 시그널 권장 비중: 시드의 {next_weight}"
            )
        else:
            st["consec_losses"] = 0
            send_tg(
                f"🎯 [{sym}] 목표 익절(TP) 도달!\n"
                f"• 방향: {pos['side']}\n"
                f"• 진입가: {format_price(sym, pos['entry'])} → 익절가: {format_price(sym, pos['tp'])} (+1.2%)\n"
                f"• 연패 초기화 완료 (다음 권장 비중: 시드의 20%)"
            )
        st["active_pos"] = None
        return False

    return True

def check_signal(symbol: str):
    df_1h = fetch_klines(symbol, "60", 80)
    ema20_1h = df_1h["close"].ewm(span=20, adjust=False).mean().iloc[-2]
    ema50_1h = df_1h["close"].ewm(span=50, adjust=False).mean().iloc[-2]
    htf_trend = "LONG (매수)" if ema20_1h > ema50_1h else "SHORT (매도)"

    df_5m = fetch_klines(symbol, "5", 80)
    candle_ts = str(df_5m["ts"].iloc[-2])
    high, low, close = df_5m["high"], df_5m["low"], df_5m["close"]

    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], 0).fillna(0)
    adx_val = float(dx.ewm(alpha=1/14, adjust=False).mean().iloc[-2])

    mid = close.rolling(20).mean().iloc[-2]
    std = close.rolling(20).std().iloc[-2]
    bbw = (4 * std) / mid if mid > 0 else 0

    if adx_val < 22.0 or bbw < 0.0045:
        return None, candle_ts

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    price = float(close.iloc[-1])
    dec = 4 if symbol == "ADAUSDT" else 2

    # 최근 3개 마감 봉(-2, -3, -4) 이내에 골든/데드크로스가 발생했는지 확인
    recent_gold = (
        (ema9.iloc[-2] > ema21.iloc[-2] and ema9.iloc[-3] <= ema21.iloc[-3]) or
        (ema9.iloc[-3] > ema21.iloc[-3] and ema9.iloc[-4] <= ema21.iloc[-4]) or
        (ema9.iloc[-4] > ema21.iloc[-4] and ema9.iloc[-5] <= ema21.iloc[-5])
    )
    recent_dead = (
        (ema9.iloc[-2] < ema21.iloc[-2] and ema9.iloc[-3] >= ema21.iloc[-3]) or
        (ema9.iloc[-3] < ema21.iloc[-3] and ema9.iloc[-4] >= ema21.iloc[-4]) or
        (ema9.iloc[-4] < ema21.iloc[-4] and ema9.iloc[-5] >= ema21.iloc[-5])
    )

    if htf_trend.startswith("LONG") and ema9.iloc[-2] > ema21.iloc[-2] and recent_gold and close.iloc[-2] > ema9.iloc[-2]:
        return ("LONG (매수)", price, round(price * 1.012, dec), round(price * 0.9905, dec), adx_val), candle_ts
    if htf_trend.startswith("SHORT") and ema9.iloc[-2] < ema21.iloc[-2] and recent_dead and close.iloc[-2] < ema9.iloc[-2]:
        return ("SHORT (매도)", price, round(price * 0.988, dec), round(price * 1.0095, dec), adx_val), candle_ts

    return None, candle_ts

def main():
    st = load_state()

    # 1. 기존 포지션이 살아있으면 청산 여부부터 추적하고, 미청산 시 신규 알림 차단
    still_open = track_active_position(st)
    if still_open:
        if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
            pos = st["active_pos"]
            send_tg(f"⏳ 현재 [{pos['symbol']}] {pos['side']} 포지션 추적 중 (진입: {format_price(pos['symbol'], pos['entry'])})")
        save_state(st)
        return

    # 2. 무포지션 상태일 때만 4종목 신규 스캔
    candidates = []
    for sym in SYMBOLS:
        res, ts = check_signal(sym)
        if res and st["last_ts"].get(sym) != ts:
            side, entry, tp, sl, adx_val = res
            candidates.append({
                "symbol": sym, "ts": ts, "side": side,
                "entry": entry, "tp": tp, "sl": sl, "adx": adx_val
            })

    if candidates:
        best = max(candidates, key=lambda x: x["adx"])
        for c in candidates:
            st["last_ts"][c["symbol"]] = c["ts"]

        st["active_pos"] = {
            "symbol": best["symbol"], "ts": best["ts"], "side": best["side"],
            "entry": best["entry"], "tp": best["tp"], "sl": best["sl"]
        }

        sym = best["symbol"]
        weight_txt = "10% (2연패 방어 모드)" if st["consec_losses"] >= 2 else "20% (기본 비중)"
        send_tg(
            f"🚨 [{sym}] 최우선 추세 돌파 시그널! (후보 {len(candidates)}개 중 ADX 1위)\n"
            f"• 방향: {best['side']}\n"
            f"• 현재가(진입): {format_price(sym, best['entry'])}\n"
            f"• 목표 익절(TP): {format_price(sym, best['tp'])} (+1.2%)\n"
            f"• 기준 손절(SL): {format_price(sym, best['sl'])} (-0.95%)\n"
            f"• 추세 강도(ADX): {best['adx']:.1f}\n"
            f"• 권장 진입 비중: 시드의 {weight_txt}"
        )
    elif os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        status_lines = []
        for sym in SYMBOLS:
            df_1h = fetch_klines(sym, "60", 80)
            e20 = df_1h["close"].ewm(span=20, adjust=False).mean().iloc[-2]
            e50 = df_1h["close"].ewm(span=50, adjust=False).mean().iloc[-2]
            trend = "상승(LONG대기)" if e20 > e50 else "하락(SHORT대기)"
            status_lines.append(f"• {sym}: {trend}")
        send_tg("✅ 4종목 시그널 봇 정상 (무포지션)\n" + "\n".join(status_lines))

    save_state(st)

if __name__ == "__main__":
    main()
