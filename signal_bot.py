import os
import json
import datetime
import numpy as np
import pandas as pd
import requests

TG_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = str(os.environ["TELEGRAM_CHAT_ID"])
STATE_FILE = "last_alert.json"

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT",
    "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT", "DOTUSDT"
]
DECIMALS = {
    "BTCUSDT": 2, "ETHUSDT": 2, "SOLUSDT": 2, "AVAXUSDT": 2, "LINKUSDT": 2,
    "DOTUSDT": 3, "ADAUSDT": 4, "XRPUSDT": 4, "SUIUSDT": 4, "DOGEUSDT": 5
}

# 현물 기준 1~2% 타깃 (익절 +1.5% / 손절 -0.9% / 최대 보유 4시간)
TP_PCT = 0.015
SL_PCT = 0.009
MAX_HOLD_MS = 4 * 60 * 60 * 1000

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
    granularity = int(interval) * 60
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
    dec = DECIMALS.get(symbol, 4)
    return f"${price:,.{dec}f}"

def track_active_position(st: dict) -> bool:
    pos = st.get("active_pos")
    if not pos:
        return False

    sym = pos["symbol"]
    df_5m = fetch_klines(sym, "5", 60)
    entry_ts = int(pos["ts"])
    latest_ts = int(df_5m["ts"].iloc[-1])

    if (latest_ts - entry_ts) >= MAX_HOLD_MS:
        send_tg(
            f"⏱️ [{sym}] 4시간 경과 시간청산 (스캔 재개)\n"
            f"• 유형: {pos['side']} | 진입가: {format_price(sym, pos['entry'])}\n"
            f"• 누적 연패: {st['consec_losses']}연패"
        )
        st["active_pos"] = None
        return False

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
            next_weight = "10% (2연패 방어)" if st["consec_losses"] >= 2 else "20% (기본)"
            send_tg(
                f"🛑 [{sym}] 손절(SL) 도달 (-{SL_PCT*100:.1f}%)\n"
                f"• 유형: {pos['side']}\n"
                f"• 진입: {format_price(sym, pos['entry'])} → 손절: {format_price(sym, pos['sl'])}\n"
                f"• 누적 연패: {st['consec_losses']}연패 | 다음 권장 비중: {next_weight}"
            )
        else:
            st["consec_losses"] = 0
            send_tg(
                f"🎯 [{sym}] 목표 익절(TP) 도달 (+{TP_PCT*100:.1f}%)\n"
                f"• 유형: {pos['side']}\n"
                f"• 진입: {format_price(sym, pos['entry'])} → 익절: {format_price(sym, pos['tp'])}\n"
                f"• 연패 초기화 (다음 권장 비중: 20%)"
            )
        st["active_pos"] = None
        return False

    return True

def calc_adx(df: pd.DataFrame) -> float:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], 0).fillna(0)
    return float(dx.ewm(alpha=1/14, adjust=False).mean().iloc[-2])

def check_signal(symbol: str):
    # 1. 상위 추세 기준: 15분봉 EMA(20/50) + ADX
    df_15m = fetch_klines(symbol, "15", 80)
    ema20_15m = df_15m["close"].ewm(span=20, adjust=False).mean().iloc[-2]
    ema50_15m = df_15m["close"].ewm(span=50, adjust=False).mean().iloc[-2]
    adx_15m = calc_adx(df_15m)
    trend_15m = "LONG" if ema20_15m > ema50_15m else "SHORT"

    # 2. 진입 트리거: 5분봉
    df_5m = fetch_klines(symbol, "5", 80)
    candle_ts = str(df_5m["ts"].iloc[-2])
    open_p, close, vol = df_5m["open"], df_5m["close"], df_5m["volume"]

    adx_5m = calc_adx(df_5m)
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - (100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)

    vol_ma = vol.iloc[-22:-2].mean()
    vol_ratio = vol.iloc[-2] / vol_ma if vol_ma > 0 else 0
    price = float(close.iloc[-1])
    dec = DECIMALS.get(symbol, 4)

    # [엔진 B] 추세 전환 (저점 반등 LONG / 고점 꺾임 SHORT)
    # 직전 3봉 내 과매도(<=28) 진입 후 30 상향 돌파 + 거래량 1.8배 + 양봉
    oversold_recent = (rsi.iloc[-4:-1].min() <= 28.0) and (rsi.iloc[-2] > 30.0)
    if oversold_recent and vol_ratio >= 1.8 and close.iloc[-2] > open_p.iloc[-2]:
        score = max(adx_5m, 22.0) + (vol_ratio * 2)
        return ("LONG (추세전환·저점반등)", price, round(price * (1 + TP_PCT), dec), round(price * (1 - SL_PCT), dec), score), candle_ts

    # 직전 3봉 내 과매수(>=72) 진입 후 70 하향 이탈 + 거래량 1.8배 + 음봉
    overbought_recent = (rsi.iloc[-4:-1].max() >= 72.0) and (rsi.iloc[-2] < 70.0)
    if overbought_recent and vol_ratio >= 1.8 and close.iloc[-2] < open_p.iloc[-2]:
        score = max(adx_5m, 22.0) + (vol_ratio * 2)
        return ("SHORT (추세전환·고점하락)", price, round(price * (1 - TP_PCT), dec), round(price * (1 + SL_PCT), dec), score), candle_ts

    # [엔진 A] 추세 유지 (15분봉 추세 + ADX 20 이상 + 5분봉 EMA 교차 + 거래량 1.3배)
    if adx_15m < 20.0 or adx_5m < 20.0 or vol_ratio < 1.3:
        return None, candle_ts

    recent_gold = any((ema9.iloc[-i] > ema21.iloc[-i] and ema9.iloc[-i-1] <= ema21.iloc[-i-1]) for i in range(2, 6))
    recent_dead = any((ema9.iloc[-i] < ema21.iloc[-i] and ema9.iloc[-i-1] >= ema21.iloc[-i-1]) for i in range(2, 6))

    if trend_15m == "LONG" and ema9.iloc[-2] > ema21.iloc[-2] and recent_gold and close.iloc[-2] > ema9.iloc[-2]:
        return ("LONG (추세유지·상승지속)", price, round(price * (1 + TP_PCT), dec), round(price * (1 - SL_PCT), dec), adx_5m), candle_ts

    if trend_15m == "SHORT" and ema9.iloc[-2] < ema21.iloc[-2] and recent_dead and close.iloc[-2] < ema9.iloc[-2]:
        return ("SHORT (추세유지·하락지속)", price, round(price * (1 - TP_PCT), dec), round(price * (1 + SL_PCT), dec), adx_5m), candle_ts

    return None, candle_ts

def main():
    st = load_state()

    if track_active_position(st):
        if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
            pos = st["active_pos"]
            send_tg(f"⏳ [{pos['symbol']}] {pos['side']} 포지션 진행 중 (진입: {format_price(pos['symbol'], pos['entry'])})")
        save_state(st)
        return

    candidates = []
    for sym in SYMBOLS:
        try:
            res, ts = check_signal(sym)
            if res and st["last_ts"].get(sym) != ts:
                side, entry, tp, sl, score = res
                candidates.append({
                    "symbol": sym, "ts": ts, "side": side,
                    "entry": entry, "tp": tp, "sl": sl, "score": score
                })
        except Exception:
            continue

    if candidates:
        best = max(candidates, key=lambda x: x["score"])
        for c in candidates:
            st["last_ts"][c["symbol"]] = c["ts"]

        st["active_pos"] = {
            "symbol": best["symbol"], "ts": best["ts"], "side": best["side"],
            "entry": best["entry"], "tp": best["tp"], "sl": best["sl"]
        }

        sym = best["symbol"]
        weight_txt = "10% (2연패 방어)" if st["consec_losses"] >= 2 else "20% (기본)"
        send_tg(
            f"🚨 [{sym}] 시그널 발생! (후보 {len(candidates)}종목 중 1위)\n"
            f"• 시그널 유형: {best['side']}\n"
            f"• 현재가(진입): {format_price(sym, best['entry'])}\n"
            f"• 목표 익절(TP): {format_price(sym, best['tp'])} (+{TP_PCT*100:.1f}%)\n"
            f"• 기준 손절(SL): {format_price(sym, best['sl'])} (-{SL_PCT*100:.1f}%)\n"
            f"• 권장 시드 비중: {weight_txt} (배수는 자율 조절)"
        )
    else:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        is_manual = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
        if is_manual or (now_utc.hour == 0 and now_utc.minute < 10):
            status_lines = []
            for sym in SYMBOLS[:4]:
                df_15m = fetch_klines(sym, "15", 80)
                e20 = df_15m["close"].ewm(span=20, adjust=False).mean().iloc[-2]
                e50 = df_15m["close"].ewm(span=50, adjust=False).mean().iloc[-2]
                status_lines.append(f"• {sym}: {'상승추세' if e20 > e50 else '하락추세'}")
            send_tg("✅ 10종목 봇 정상 가동 중 (추세유지 + 추세전환 감시)\n" + "\n".join(status_lines))

    save_state(st)

if __name__ == "__main__":
    main()
