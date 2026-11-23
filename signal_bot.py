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
MAX_HOLD_MS = 6 * 60 * 60 * 1000  # 6시간 이상 횡보 시 포지션 자동 해제

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
    dec = DECIMALS.get(symbol, 4)
    return f"${price:,.{dec}f}"

def track_active_position(st: dict) -> bool:
    pos = st.get("active_pos")
    if not pos:
        return False

    sym = pos["symbol"]
    df_5m = fetch_klines(sym, "5", 80)
    entry_ts = int(pos["ts"])
    latest_ts = int(df_5m["ts"].iloc[-1])

    if (latest_ts - entry_ts) >= MAX_HOLD_MS:
        send_tg(
            f"⏱️ [{sym}] 6시간 경과 횡보 종료 (시간청산)\n"
            f"• 방향: {pos['side']} | 진입가: {format_price(sym, pos['entry'])}\n"
            f"• 신규 시그널 스캔 재개 (연패 유지: {st['consec_losses']}연패)"
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
    high, low, close, vol = df_5m["high"], df_5m["low"], df_5m["close"], df_5m["volume"]

    # ADX & BBW 계산
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

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    price = float(close.iloc[-1])
    dec = DECIMALS.get(symbol, 4)

    # 트랙 2: 저점 과매도 반등 (RSI 30 상향 돌파 + 거래량 2배 급증 + 양봉)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - (100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)
    vol_spike = vol.iloc[-2] > (vol.iloc[-22:-2].mean() * 2.0)

    bounce_long = (rsi.iloc[-3] <= 30.0) and (rsi.iloc[-2] > 30.0) and vol_spike and (close.iloc[-2] > df_5m["open"].iloc[-2])
    if bounce_long:
        return ("LONG (저점반등)", price, round(price * 1.012, dec), round(price * 0.9905, dec), max(adx_val, 22.0)), candle_ts

    # 트랙 1: 기존 추세 돌파 (1H 정배열/역배열 + ADX 22 이상 + BBW 0.0045 이상 + 최근 6봉 교차)
    if adx_val < 22.0 or bbw < 0.0045:
        return None, candle_ts

    recent_gold = any((ema9.iloc[-i] > ema21.iloc[-i] and ema9.iloc[-i-1] <= ema21.iloc[-i-1]) for i in range(2, 8))
    recent_dead = any((ema9.iloc[-i] < ema21.iloc[-i] and ema9.iloc[-i-1] >= ema21.iloc[-i-1]) for i in range(2, 8))

    if htf_trend.startswith("LONG") and ema9.iloc[-2] > ema21.iloc[-2] and recent_gold and close.iloc[-2] > ema9.iloc[-2]:
        return ("LONG (추세돌파)", price, round(price * 1.012, dec), round(price * 0.9905, dec), adx_val), candle_ts
    if htf_trend.startswith("SHORT") and ema9.iloc[-2] < ema21.iloc[-2] and recent_dead and close.iloc[-2] < ema9.iloc[-2]:
        return ("SHORT (추세이탈)", price, round(price * 0.988, dec), round(price * 1.0095, dec), adx_val), candle_ts

    return None, candle_ts

def main():
    st = load_state()

    still_open = track_active_position(st)
    if still_open:
        if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
            pos = st["active_pos"]
            send_tg(f"⏳ 현재 [{pos['symbol']}] {pos['side']} 포지션 추적 중 (진입: {format_price(pos['symbol'], pos['entry'])})")
        save_state(st)
        return

    candidates = []
    for sym in SYMBOLS:
        try:
            res, ts = check_signal(sym)
            if res and st["last_ts"].get(sym) != ts:
                side, entry, tp, sl, adx_val = res
                candidates.append({
                    "symbol": sym, "ts": ts, "side": side,
                    "entry": entry, "tp": tp, "sl": sl, "adx": adx_val
                })
        except Exception:
            continue

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
            f"🚨 [{sym}] 최우선 시그널 포착! (후보 {len(candidates)}개 중 ADX 1위)\n"
            f"• 유형: {best['side']}\n"
            f"• 현재가(진입): {format_price(sym, best['entry'])}\n"
            f"• 목표 익절(TP): {format_price(sym, best['tp'])} (+1.2%)\n"
            f"• 기준 손절(SL): {format_price(sym, best['sl'])} (-0.95%)\n"
            f"• 추세 강도(ADX): {best['adx']:.1f}\n"
            f"• 권장 진입 비중: 시드의 {weight_txt}"
        )
    else:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        is_manual = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
        if is_manual or (now_utc.hour == 0 and now_utc.minute < 10):
            status_lines = []
            for sym in SYMBOLS[:4]:
                df_1h = fetch_klines(sym, "60", 80)
                e20 = df_1h["close"].ewm(span=20, adjust=False).mean().iloc[-2]
                e50 = df_1h["close"].ewm(span=50, adjust=False).mean().iloc[-2]
                trend = "상승(LONG대기)" if e20 > e50 else "하락(SHORT대기)"
                status_lines.append(f"• {sym}: {trend}")
            send_tg("✅ 10종목 시그널 봇 정상 (무포지션 · 추세+저점반등 감시 중)\n" + "\n".join(status_lines))

    save_state(st)

if __name__ == "__main__":
    main()
