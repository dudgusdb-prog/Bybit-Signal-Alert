import os
import json
import numpy as np
import pandas as pd
import requests

TG_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = str(os.environ["TELEGRAM_CHAT_ID"])
STATE_FILE = "last_alert.json"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

def send_tg(text: str):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    resp = requests.post(url, json={"chat_id": CHAT_ID, "text": text}, timeout=10)
    resp.raise_for_status()

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"last_ts": {}}

def save_state(st: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=2)

def fetch_klines(symbol: str, interval: str, limit: int = 80) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    # 1순위: OKX 무기한 선물 공개 API (미국 GitHub 서버 차단 없음)
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

    # 2순위: Coinbase 공개 캔들 API (백업)
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

def check_signal(symbol: str):
    df_1h = fetch_klines(symbol, "60", 80)
    ema20_1h = df_1h["close"].ewm(span=20, adjust=False).mean().iloc[-2]
    ema50_1h = df_1h["close"].ewm(span=50, adjust=False).mean().iloc[-2]
    htf_trend = "LONG (매수)" if ema20_1h > ema50_1h else "SHORT (매도)"

    df_5m = fetch_klines(symbol, "5", 80)
    candle_ts = str(df_5m["ts"].iloc[-2])
    high, low, close = df_5m["high"], df_5m["low"], df_5m["close"]

    # ADX(14) 계산
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

    # BBW(20) 계산
    mid = close.rolling(20).mean().iloc[-2]
    std = close.rolling(20).std().iloc[-2]
    bbw = (4 * std) / mid if mid > 0 else 0

    # 횡보장 필터
    if adx_val < 22.0 or bbw < 0.0045:
        return None, candle_ts

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    price = float(close.iloc[-1])

    if htf_trend.startswith("LONG") and ema9.iloc[-2] > ema21.iloc[-2] and ema9.iloc[-3] <= ema21.iloc[-3] and close.iloc[-2] > ema9.iloc[-2]:
        return ("LONG (매수)", price, round(price * 1.012, 2), round(price * 0.9905, 2), adx_val), candle_ts
    if htf_trend.startswith("SHORT") and ema9.iloc[-2] < ema21.iloc[-2] and ema9.iloc[-3] >= ema21.iloc[-3] and close.iloc[-2] < ema9.iloc[-2]:
        return ("SHORT (매도)", price, round(price * 0.988, 2), round(price * 1.0095, 2), adx_val), candle_ts

    return None, candle_ts

def main():
    st = load_state()
    sent_any = False
    for sym in SYMBOLS:
        res, ts = check_signal(sym)
        if res and st["last_ts"].get(sym) != ts:
            side, entry, tp, sl, adx_val = res
            st["last_ts"][sym] = ts
            sent_any = True
            send_tg(
                f"🚨 [{sym}] 추세 돌파 시그널 발생!\n"
                f"• 방향: {side}\n"
                f"• 현재가(진입): ${entry:,.2f}\n"
                f"• 목표 익절(TP): ${tp:,.2f} (+1.2%)\n"
                f"• 기준 손절(SL): ${sl:,.2f} (-0.95%)\n"
                f"• 추세 강도(ADX): {adx_val:.1f}"
            )

    # 수동 실행(Run workflow)일 때만 텔레그램 연결 테스트 메시지 발송
    if not sent_any and os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        send_tg("✅ 시그널 봇 연결 정상! (현재는 횡보장이라 진입 타점 없음)")

    save_state(st)

if __name__ == "__main__":
    main()
