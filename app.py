import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import io
import time
import math
import logging
from bs4 import BeautifulSoup
import FinanceDataReader as fdr
from google import genai

# =============================================================
# [시스템 로깅 & 환경 설정]
# =============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("QuantExecutionProMax")

MODEL_CANDIDATES = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]

# 2023년 개정 KRX 공식 호가 단위 (p <= threshold 기준 경계값 일치)
KOREA_TICK_TABLE = [
    (2000, 1),
    (5000, 5),
    (20000, 10),
    (50000, 20),
    (200000, 100),
    (500000, 200),
    (float('inf'), 1000)
]

st.set_page_config(
    page_title="QUANT-EXECUTION PRO MAX: 완전 일치 실행 엔진",
    page_icon="⚖️",
    layout="wide"
)

# -------------------------------------------------------------
# 0. API 설정 & AI 설명 데스크
# -------------------------------------------------------------
secret_key = ""
try:
    if "GEMINI_API_KEY" in st.secrets:
        secret_key = st.secrets["GEMINI_API_KEY"].strip()
except Exception:
    pass

st.sidebar.header("⚙️ 시스템 설정")
input_key = st.sidebar.text_input("Gemini API Key", value=secret_key, type="password")
active_key = input_key.strip() if input_key.strip() else secret_key

client = None
if active_key:
    try:
        client = genai.Client(api_key=active_key)
        st.sidebar.success("✅ AI 리스크 설명 데스크 활성화")
    except Exception as e:
        client = None
        st.sidebar.warning(f"AI 비활성화 (순수 규칙 엔진만 가동): {e}")

def generate_ai_briefing(client, prompt):
    if not client: return ""
    for model_name in MODEL_CANDIDATES:
        try:
            res = client.models.generate_content(model=model_name, contents=prompt)
            if res and res.text: return res.text
        except Exception:
            continue
    return "AI 브리핑을 생성할 수 없습니다."

# -------------------------------------------------------------
# 1. 호가 단위 및 가격 헬퍼
# -------------------------------------------------------------
def get_tick_size(price):
    try:
        p = abs(float(price))
        for threshold, tick in KOREA_TICK_TABLE:
            if p <= threshold:
                return tick
    except (ValueError, TypeError):
        return 1
    return 1000

def floor_to_tick(price):
    if price is None or pd.isna(price) or float(price) <= 0: return 0
    p = float(price)
    t = get_tick_size(p)
    return int(math.floor(p / t) * t)

def ceil_to_tick(price):
    if price is None or pd.isna(price) or float(price) <= 0: return 0
    p = float(price)
    t = get_tick_size(p)
    return int(math.ceil(p / t) * t)

# -------------------------------------------------------------
# 2. 보조 지표 계산 함수군 (Wilder ATR & 정제 거래대금)
# -------------------------------------------------------------
def calculate_wilder_atr(df, period=14):
    if len(df) < period + 1: return 0.0
    high, low, close_prev = df['High'], df['Low'], df['Close'].shift(1)
    tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else 0.0

def get_clean_avg_volume(volume_series, lookback=20):
    past_vols = volume_series.iloc[-(lookback * 2 + 1):-1]
    valid_vols = past_vols[past_vols > 0]
    if len(valid_vols) >= lookback:
        return float(valid_vols.tail(lookback).mean())
    elif len(valid_vols) > 0:
        return float(valid_vols.mean())
    return 1.0

def get_clean_avg_amount(df_slice, lookback=20):
    if 'Amount' in df_slice.columns and not df_slice['Amount'].iloc[-lookback*2:-1].isnull().all() and (df_slice['Amount'].iloc[-lookback*2:-1] > 0).any():
        amt_series = df_slice['Amount'].iloc[-lookback*2:-1]
    else:
        amt_series = (df_slice['Close'] * df_slice['Volume']).iloc[-lookback*2:-1]
    valid_amt = amt_series[amt_series > 0]
    if len(valid_amt) >= lookback:
        return float(valid_amt.tail(lookback).mean())
    elif len(valid_amt) > 0:
        return float(valid_amt.mean())
    return 1.0

# -------------------------------------------------------------
# 3. KOSPI + KOSDAQ 듀얼 과거 Regime 시계열 엔진
# -------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_historical_dual_regime(start_date="2022-01-01"):
    try:
        df_ks = fdr.DataReader('KS11', start_date)
        df_kq = fdr.DataReader('KQ11', start_date)
        if df_ks is None or df_kq is None or len(df_ks) < 30:
            return pd.DataFrame()

        df_ks.index = pd.to_datetime(df_ks.index).normalize()
        df_kq.index = pd.to_datetime(df_kq.index).normalize()

        common_idx = df_ks.index.intersection(df_kq.index)
        df_ks = df_ks.loc[common_idx]
        df_kq = df_kq.loc[common_idx]

        for df in [df_ks, df_kq]:
            df['MA20'] = df['Close'].rolling(20).mean()
            df['Slope'] = ((df['MA20'] - df['MA20'].shift(5)) / df['MA20'].shift(5)) * 100

        regimes = []
        for idx in range(len(df_ks)):
            c_ks, m_ks, s_ks = df_ks['Close'].iloc[idx], df_ks['MA20'].iloc[idx], df_ks['Slope'].iloc[idx]
            c_kq, m_kq, s_kq = df_kq['Close'].iloc[idx], df_kq['MA20'].iloc[idx], df_kq['Slope'].iloc[idx]

            ks_bull = (c_ks >= m_ks) and (s_ks > 0)
            kq_bull = (c_kq >= m_kq) and (s_kq > 0)
            ks_bear = (c_ks < m_ks) and (s_ks <= 0)
            kq_bear = (c_kq < m_kq) and (s_kq <= 0)

            if ks_bear or kq_bear:
                regimes.append("RISK-OFF")
            elif ks_bull and kq_bull:
                regimes.append("FAVORABLE")
            else:
                regimes.append("NEUTRAL")

        df_res = pd.DataFrame(index=common_idx)
        df_res['Regime'] = regimes
        df_res['KS_Close'] = df_ks['Close']
        df_res['KQ_Close'] = df_kq['Close']
        return df_res
    except Exception as e:
        logger.error(f"듀얼 Regime 로드 실패: {e}")
        return pd.DataFrame()

def get_current_dual_regime():
    df_dual = load_historical_dual_regime((datetime.date.today() - datetime.timedelta(days=90)).strftime('%Y-%m-%d'))
    if df_dual.empty:
        return {"status": "NEUTRAL", "code": "NEUTRAL", "multiplier": 0.5, "msg": "시장 지수 수신 지연"}
    
    last = df_dual.iloc[-1]
    regime = last['Regime']
    if regime == "FAVORABLE":
        return {"status": "FAVORABLE (양대 지수 우호)", "code": "FAVORABLE", "multiplier": 1.0, "msg": "KOSPI/KOSDAQ 20일선 상회 및 상승세. 정상 매수 집행."}
    elif regime == "RISK-OFF":
        return {"status": "RISK-OFF (시장 위험 경보)", "code": "RISK-OFF", "multiplier": 0.0, "msg": "주요 지수 20일선 하회 및 하락세. 신규 매수를 전면 금지합니다."}
    else:
        return {"status": "NEUTRAL (경계/혼조 국면)", "code": "NEUTRAL", "multiplier": 0.5, "msg": "양대 시장 혼조세. 신규 매수 시 비중을 50%로 축소 집행합니다."}

# -------------------------------------------------------------
# 4. 데이터 로더 & 과거 외국인 수급 일별 시계열 크롤러
# -------------------------------------------------------------
@st.cache_data(ttl=86400)
def load_all_krx_master():
    try:
        url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        if res.status_code == 200:
            dfs = pd.read_html(io.BytesIO(res.content), encoding="cp949")
            if dfs and not dfs[0].empty:
                df = dfs[0][['회사명', '종목코드']].copy()
                df.columns = ['Name', 'Code']
                df['Code'] = df['Code'].astype(str).str.zfill(6)
                return df.drop_duplicates(subset=['Code']).reset_index(drop=True)
    except Exception as e:
        logger.error(f"KIND 로드 실패: {e}")
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_krx_listing():
    try:
        df = fdr.StockListing('KRX')
        if df is not None and not df.empty and 'Code' in df.columns:
            df['Code'] = df['Code'].astype(str).str.zfill(6)
            for num_col in ['Close', 'Volume', 'Amount', 'Marcap']:
                if num_col in df.columns:
                    df[num_col] = pd.to_numeric(df[num_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            if 'Amount' not in df.columns or (df['Amount'] == 0).all():
                df['Amount'] = df['Close'] * df['Volume']
            return df.drop_duplicates(subset=['Code']).reset_index(drop=True)
    except Exception as e:
        logger.error(f"KRX 로드 실패: {e}")
    return pd.DataFrame()

def find_stock_code(query, df_krx):
    query = query.strip()
    if not query: return None, None
    if query.isdigit() and len(query) == 6:
        master = load_all_krx_master()
        if not master.empty:
            m = master[master['Code'] == query]
            if not m.empty: return query, m.iloc[0]['Name']
        return query, query

    master = load_all_krx_master()
    if not master.empty:
        matched = master[master['Name'].astype(str).str.upper().str.replace(" ", "") == query.upper().replace(" ", "")]
        if not matched.empty: return matched.iloc[0]['Code'], matched.iloc[0]['Name']

    if df_krx is not None and not df_krx.empty:
        matched = df_krx[df_krx['Name'].astype(str).str.upper().str.replace(" ", "") == query.upper().replace(" ", "")]
        if not matched.empty: return matched.iloc[0]['Code'], matched.iloc[0]['Name']
    return None, None

def get_supply_data(code):
    headers = {"User-Agent": "Mozilla/5.0"}
    res_data = {"frgn_5": 0, "inst_5": 0, "status": "FAIL"}
    try:
        url = f"https://finance.naver.com/item/frgn.naver?code={code}"
        res = requests.get(url, headers=headers, timeout=4)
        if res.status_code == 200:
            dfs = pd.read_html(io.BytesIO(res.content), encoding="cp949")
            df_frgn = next((t for t in dfs if "기관" in str(t.columns) and "외국인" in str(t.columns)), None)
            if df_frgn is not None:
                df_frgn.columns = ['_'.join(c).strip() if isinstance(c, tuple) else str(c) for c in df_frgn.columns]
                i_cols = [c for c in df_frgn.columns if "기관" in c and "순매" in c]
                f_cols = [c for c in df_frgn.columns if "외국인" in c and "순매" in c]
                if i_cols and f_cols:
                    iv = pd.to_numeric(df_frgn[i_cols[0]].astype(str).str.replace(',', ''), errors='coerce').dropna().tolist()
                    fv = pd.to_numeric(df_frgn[f_cols[0]].astype(str).str.replace(',', ''), errors='coerce').dropna().tolist()
                    if len(iv) >= 5 and len(fv) >= 5:
                        res_data["inst_5"] = int(sum(iv[:5]))
                        res_data["frgn_5"] = int(sum(fv[:5]))
                        res_data["status"] = "OK"
    except Exception:
        pass
    return res_data

@st.cache_data(ttl=86400)
def load_historical_foreign_series(code, pages=18):
    """
    네이버 금융 일별 외국인 순매수 시계열 수신 (~360거래일, 약 1.5년치)
    백테스트 시점의 과거 5일 외국인 순매수 합계를 100% 동일하게 복원
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    records = []
    for p in range(1, pages + 1):
        try:
            url = f"https://finance.naver.com/item/frgn.naver?code={code}&page={p}"
            res = requests.get(url, headers=headers, timeout=3)
            if res.status_code != 200: break
            dfs = pd.read_html(io.BytesIO(res.content), encoding="cp949")
            df_frgn = next((t for t in dfs if "기관" in str(t.columns) and "외국인" in str(t.columns)), None)
            if df_frgn is not None:
                df_frgn.columns = ['_'.join(c).strip() if isinstance(c, tuple) else str(c) for c in df_frgn.columns]
                date_col = [c for c in df_frgn.columns if "날짜" in c]
                frgn_col = [c for c in df_frgn.columns if "외국인" in c and "순매" in c]
                if date_col and frgn_col:
                    sub = df_frgn[[date_col[0], frgn_col[0]]].dropna()
                    sub.columns = ['Date', 'Net_Frgn']
                    sub['Date'] = pd.to_datetime(sub['Date'], errors='coerce')
                    sub['Net_Frgn'] = pd.to_numeric(sub['Net_Frgn'].astype(str).str.replace(',', ''), errors='coerce')
                    records.append(sub.dropna())
        except Exception:
            break
    if not records: return pd.Series(dtype=float)
    total_df = pd.concat(records).drop_duplicates(subset=['Date']).sort_values('Date').set_index('Date')
    total_df.index = pd.to_datetime(total_df.index).normalize()
    # 과거 5거래일 순매수 합계
    frgn_5d_sum = total_df['Net_Frgn'].rolling(5).sum()
    return frgn_5d_sum

# -------------------------------------------------------------
# 5. [100점 Entry Score] 20거래일 인덱스 보정 엔진
# -------------------------------------------------------------
def calculate_100p_score(df_slice, kospi_slice=None, has_frgn=False):
    if len(df_slice) < 25:
        return {"score": 0, "grade": "AVOID", "details": {}, "track": "WAIT"}

    curr_p = float(df_slice['Close'].iloc[-1])
    open_p = float(df_slice['Open'].iloc[-1])
    high_p = float(df_slice['High'].iloc[-1])
    low_p = float(df_slice['Low'].iloc[-1])

    ma5 = float(df_slice['Close'].rolling(5).mean().iloc[-1])
    ma20 = float(df_slice['Close'].rolling(20).mean().iloc[-1])
    ma20_5d = float(df_slice['Close'].rolling(20).mean().iloc[-6])
    ma60 = float(df_slice['Close'].rolling(60).mean().iloc[-1]) if len(df_slice) >= 60 else ma20

    ma20_slope_pct = ((ma20 - ma20_5d) / ma20_5d) * 100 if ma20_5d > 0 else 0.0
    disparity_pct = ((curr_p - ma20) / ma20) * 100

    score = 0
    details = {}

    # 1. 추세 구조 (20점)
    trend_pts = 0
    if curr_p >= ma20 >= ma60: trend_pts += 12
    elif curr_p >= ma20: trend_pts += 8
    if ma20_slope_pct > 0.3: trend_pts += 8
    elif ma20_slope_pct > 0: trend_pts += 5
    score += trend_pts
    details["추세구조"] = f"{trend_pts}/20"

    # 2. MA20 지지/이격 (20점)
    disp_pts = 0
    if -1.0 <= disparity_pct <= 2.5: disp_pts += 20
    elif 2.5 < disparity_pct <= 5.0: disp_pts += 12
    elif -2.5 <= disparity_pct < -1.0: disp_pts += 8
    score += disp_pts
    details["MA20지지"] = f"{disp_pts}/20"

    # 3. 모멘텀 & 캔들 장악력 (15점)
    c_pts = 0
    c_range = (high_p - low_p) if high_p > low_p else 1.0
    c_pos = (curr_p - low_p) / c_range
    if curr_p > open_p and c_pos >= 0.65: c_pts += 10
    elif curr_p > open_p: c_pts += 5
    if curr_p >= ma5: c_pts += 5
    score += c_pts
    details["모멘텀캔들"] = f"{c_pts}/15"

    # 4. 거래량 유입 (15점)
    vol_pts = 0
    clean_v20 = get_clean_avg_volume(df_slice['Volume'], 20)
    vol_ratio = float(df_slice['Volume'].iloc[-1] / clean_v20)
    if vol_ratio >= 1.5: vol_pts += 15
    elif vol_ratio >= 1.2: vol_pts += 10
    elif vol_ratio >= 0.9: vol_pts += 5
    score += vol_pts
    details["거래량"] = f"{vol_pts}/15"

    # 5. 상대강도 RS (15점) - 정확한 20거래일 전(iloc[-21]) 대비
    rs_pts = 5
    if len(df_slice) >= 21:
        p_prev20 = float(df_slice['Close'].iloc[-21])
        stk_ret20 = ((curr_p - p_prev20) / p_prev20) * 100
        if kospi_slice is not None and len(kospi_slice) >= 21:
            k_curr = float(kospi_slice['Close'].iloc[-1])
            k_prev20 = float(kospi_slice['Close'].iloc[-21])
            mkt_ret20 = ((k_curr - k_prev20) / k_prev20) * 100
            diff_rs = stk_ret20 - mkt_ret20
            if diff_rs >= 5.0: rs_pts = 15
            elif diff_rs >= 0.0: rs_pts = 10
            elif diff_rs >= -5.0: rs_pts = 5
            else: rs_pts = 0
        else:
            if stk_ret20 > 5.0: rs_pts = 12
    score += rs_pts
    details["상대강도"] = f"{rs_pts}/15"

    # 6. 전고점 근접 / 수급 참여 (15점)
    brk_pts = 0
    high_20 = float(df_slice['High'].iloc[-21:-1].max())
    dist_high = ((high_20 - curr_p) / curr_p) * 100
    if dist_high <= 3.0: brk_pts += 10
    elif dist_high <= 6.0: brk_pts += 5
    if has_frgn: brk_pts += 5
    score += min(15, brk_pts)
    details["돌파수급"] = f"{min(15, brk_pts)}/15"

    if score >= 82: grade = "STRONG_BUY"
    elif score >= 70: grade = "BUY"
    elif score >= 60: grade = "WATCH"
    elif score >= 50: grade = "WAIT"
    else: grade = "AVOID"

    # 3-Track 의사결정
    if grade in ["STRONG_BUY", "BUY"]:
        if -1.0 <= disparity_pct <= 1.5:
            track = "MARKET"
        elif 1.5 < disparity_pct <= 3.5:
            track = "PULLBACK"
        elif dist_high <= 2.5 and vol_ratio >= 1.3:
            track = "BREAKOUT"
        else:
            track = "MARKET"
    else:
        track = "WAIT"

    return {
        "score": score, "grade": grade, "details": details,
        "track": track, "disparity_pct": disparity_pct,
        "high_20": int(high_20), "ma20": int(ma20), "ma20_slope_pct": ma20_slope_pct
    }

# -------------------------------------------------------------
# 6. 가격선 설계 & 과도 손절폭 필터
# -------------------------------------------------------------
def calculate_execution_levels(df_slice, entry_price):
    atr14 = calculate_wilder_atr(df_slice, 14)
    curr_p = float(entry_price)
    ma20 = float(df_slice['Close'].rolling(20).mean().iloc[-1])
    recent_swing_low = float(df_slice['Low'].iloc[-10:-1].min())
    high_20 = float(df_slice['High'].iloc[-21:-1].max())

    stop_atr = curr_p - (1.5 * atr14)
    stop_struct = recent_swing_low - (0.2 * atr14)
    stop_trend = ma20 * 0.97
    candidate_stop = min(stop_atr, stop_struct, stop_trend)

    min_dist_price = curr_p - (1.2 * atr14)
    final_stop_val = min(candidate_stop, min_dist_price)
    
    stop_price = floor_to_tick(final_stop_val)
    if stop_price >= curr_p:
        stop_price = floor_to_tick(curr_p * 0.96)

    risk_1r = curr_p - stop_price
    loss_pct = round(((stop_price - curr_p) / curr_p) * 100, 2)
    is_excessive_stop = abs(loss_pct) >= 8.5

    target_1r = ceil_to_tick(curr_p + 1.0 * risk_1r)
    target_2r = ceil_to_tick(curr_p + 2.0 * risk_1r)
    target_3r = ceil_to_tick(curr_p + 3.0 * risk_1r)

    reward_to_resist = high_20 - curr_p
    rrr_to_resist = round(reward_to_resist / risk_1r, 2) if risk_1r > 0 else 0.0
    resistance_warning = None
    if target_2r > high_20:
        resistance_warning = f"2R 도달 전 직전 20일 전고점({int(high_20):,}원, 실질 RRR: {rrr_to_resist}R) 매물대 존재"

    return {
        "entry_price": int(curr_p),
        "stop_price": stop_price,
        "loss_pct": loss_pct,
        "is_excessive_stop": is_excessive_stop,
        "risk_1r": int(risk_1r),
        "target_1r": target_1r,
        "target_2r": target_2r,
        "target_3r": target_3r,
        "high_20": int(high_20),
        "rrr_to_resist": rrr_to_resist,
        "resistance_warning": resistance_warning,
        "atr14": int(atr14)
    }

def evaluate_full_execution_pipeline(df_slice, regime_code, kospi_slice=None, has_frgn=False, min_score=70):
    scored = calculate_100p_score(df_slice, kospi_slice, has_frgn=has_frgn)
    curr_p = int(df_slice['Close'].iloc[-1])
    levels = calculate_execution_levels(df_slice, curr_p)

    if scored["score"] < min_score:
        return {"action": "WAIT", "badge": f"🟡 {scored['grade']} ({scored['score']}점)", "reason": f"기준 점수({min_score}점) 미달", "scored": scored, "levels": levels, "multiplier": 0.0}

    if levels["is_excessive_stop"]:
        return {"action": "WAIT", "badge": "🟡 WAIT (손절폭 과도)", "reason": f"지지선 기반 손절폭({levels['loss_pct']}%)이 -8.5%를 초과하여 자본 회전 효율이 낮습니다.", "scored": scored, "levels": levels, "multiplier": 0.0}

    if regime_code == "RISK-OFF":
        return {"action": "AVOID", "badge": "🔴 AVOID (시장 위험 차단)", "reason": "시장 Regime이 RISK-OFF 국면이므로 모든 신규 매수를 금지합니다.", "scored": scored, "levels": levels, "multiplier": 0.0}
    elif regime_code == "NEUTRAL":
        return {"action": "BUY_REDUCED", "badge": f"🟡 BUY (비중 50% 축소 | {scored['score']}점)", "reason": "시장 중립 국면. 자본 리스크 통제를 위해 수량을 50%로 축소 집행합니다.", "scored": scored, "levels": levels, "multiplier": 0.5}
    else:
        return {"action": "BUY", "badge": f"🟢 BUY (정상 진입 | {scored['score']}점)", "reason": f"시장 우호 + 100점 만점 통과({scored['score']}점). 정상 진입을 승인합니다.", "scored": scored, "levels": levels, "multiplier": 1.0}

def calculate_position_sizing(account_capital, risk_pct, exec_multiplier, curr_price, risk_1r, atr14, avg_amount_20):
    effective_risk_pct = (risk_pct / 100.0) * exec_multiplier
    max_risk_amount = account_capital * effective_risk_pct

    if risk_1r <= 0 or curr_price <= 0 or effective_risk_pct <= 0:
        return {"shares": 0, "total_invest": 0, "weight_pct": 0.0, "normal_risk_loss": 0, "worst_gap_loss": 0, "limiting_factor": "NO_RISK_BUDGET", "real_risk_pct": 0.0}

    raw_shares = int(max_risk_amount // risk_1r)
    max_liq_shares = int((avg_amount_20 * 0.01) // curr_price) if avg_amount_20 > 0 else raw_shares
    max_weight_shares = int((account_capital * 0.30) // curr_price)

    final_shares = max(0, min(raw_shares, max_liq_shares, max_weight_shares))
    
    if final_shares == raw_shares:
        limiting_factor = "RISK_BUDGET (1% 손실한도)"
    elif final_shares == max_weight_shares:
        limiting_factor = "WEIGHT_CAP (30% 계좌한도)"
    else:
        limiting_factor = "LIQUIDITY_CAP (1% 거래대금)"

    total_invest = final_shares * curr_price
    weight_pct = round((total_invest / account_capital) * 100, 1) if account_capital > 0 else 0.0
    normal_risk_loss = final_shares * risk_1r
    worst_gap_loss = final_shares * (risk_1r + int(1.0 * atr14))
    real_risk_pct = round((normal_risk_loss / account_capital) * 100, 2) if account_capital > 0 else 0.0

    return {
        "shares": final_shares, "total_invest": int(total_invest),
        "weight_pct": weight_pct, "normal_risk_loss": int(normal_risk_loss),
        "worst_gap_loss": int(worst_gap_loss),
        "limiting_factor": limiting_factor,
        "real_risk_pct": real_risk_pct
    }

# -------------------------------------------------------------
# 7. [공통 코어] 3-Track 체결 및 전면 슬리피지/체결모드 일중 관리 엔진
# -------------------------------------------------------------
def check_3track_execution(track, b_open, b_high, b_low, ma20_entry, high_20, slippage_rate=0.0010):
    if b_open <= 0: return False, 0.0

    if track == "MARKET":
        if b_open <= ma20_entry * 1.030:
            exec_p = b_open * (1.0 + slippage_rate)
            return True, ceil_to_tick(exec_p)
    elif track == "PULLBACK":
        limit_p = floor_to_tick(ma20_entry * 1.005)
        if b_open <= limit_p:
            return True, b_open
        elif b_low <= limit_p:
            return True, limit_p
    elif track == "BREAKOUT":
        trigger_p = ceil_to_tick(high_20 * 1.002)
        if b_open >= trigger_p:
            exec_p = b_open * (1.0 + slippage_rate)
            return True, ceil_to_tick(exec_p)
        elif b_high >= trigger_p:
            exec_p = trigger_p * (1.0 + slippage_rate)
            return True, ceil_to_tick(exec_p)
    return False, 0.0

def process_intraday_position(pos, bar, ma5_val, prev_low, cur_date_str, max_holding=10, sell_fee=0.0020, 
                              market_slippage=0.0010, limit_slippage=0.0005, execution_mode="CONSERVATIVE"):
    b_open = float(bar['Open'])
    b_high = float(bar['High'])
    b_low = float(bar['Low'])
    b_close = float(bar['Close'])

    pos["holding_days"] += 1
    h_days = pos["holding_days"]
    e_price = pos["entry_price"]
    c_stop = pos["current_stop"]
    cur_shares = pos["shares"]
    pos["peak_price"] = max(pos["peak_price"], b_high)

    exit_event = None
    exit_price = 0.0
    cash_gain_today = 0.0
    is_ambiguous = False

    hit_target_candidate = (b_high >= pos["t1"])
    hit_stop_candidate = (b_low <= c_stop)

    if hit_target_candidate and hit_stop_candidate:
        is_ambiguous = True
        if execution_mode == "CONSERVATIVE":
            prioritize_stop = True
        elif execution_mode == "OPTIMISTIC":
            prioritize_stop = False
        else:  # NEUTRAL: 음봉이면 손절 우선, 양봉이면 익절 우선
            prioritize_stop = (b_close < b_open)
    else:
        prioritize_stop = hit_stop_candidate

    # 1. 오버나이트 갭다운 손절 (시장가 슬리피지 차감)
    if h_days > 1 and b_open <= c_stop:
        exit_event = "GAP_DOWN_STOP" if not pos["t1_hit"] else "GAP_DOWN_TRAILING"
        exit_price = floor_to_tick(b_open * (1.0 - market_slippage))
    # 2. 장중 손절 터치
    elif prioritize_stop and hit_stop_candidate:
        exit_event = "STOP_LOSS" if not pos["t1_hit"] else "TRAILING_STOP"
        exit_price = floor_to_tick(c_stop * (1.0 - market_slippage))
    # 3. Trend Break (시장가 슬리피지 차감)
    elif pos["t1_hit"] and (b_close < ma5_val) and (b_close < prev_low):
        exit_event = "TREND_BREAK_EXIT"
        exit_price = floor_to_tick(b_close * (1.0 - market_slippage))
    else:
        # 4. T1 도달 (30% - 지정가 슬리피지 차감)
        if not pos["t1_hit"] and b_high >= pos["t1"]:
            pos["t1_hit"] = True
            s_qty = min(int(pos["orig_shares"] * 0.30), cur_shares)
            t1_exec_p = floor_to_tick(pos["t1"] * (1.0 - limit_slippage))
            gain = s_qty * t1_exec_p * (1.0 - sell_fee)
            cash_gain_today += gain
            pos["cash_inflow"] += gain
            pos["shares"] -= s_qty
            cur_shares = pos["shares"]
            pos["current_stop"] = max(pos["current_stop"], e_price)

            # 당일 본전 급락 재검사
            if b_low <= e_price and cur_shares > 0:
                exit_event = "T1_THEN_BREAKEVEN"
                exit_price = floor_to_tick(e_price * (1.0 - market_slippage))

        # 5. 1.5R 보호 스탑 (+0.5R)
        if exit_event is None and pos["t1_hit"] and not pos["t2_hit"]:
            r1 = pos["t1"] - e_price
            if b_high >= e_price + (1.5 * r1):
                pos["current_stop"] = max(pos["current_stop"], floor_to_tick(e_price + 0.5 * r1))

        # 6. T2 도달 (40% - 지정가 슬리피지 차감) & +1.0R 보호 스탑
        if exit_event is None and not pos["t2_hit"] and b_high >= pos["t2"]:
            pos["t2_hit"] = True
            s_qty = min(int(pos["orig_shares"] * 0.40), pos["shares"])
            t2_exec_p = floor_to_tick(pos["t2"] * (1.0 - limit_slippage))
            gain = s_qty * t2_exec_p * (1.0 - sell_fee)
            cash_gain_today += gain
            pos["cash_inflow"] += gain
            pos["shares"] -= s_qty
            cur_shares = pos["shares"]
            r1 = pos["t1"] - e_price
            pos["current_stop"] = max(pos["current_stop"], floor_to_tick(e_price + 1.0 * r1))

        # 7. T2 달성 후 고점 대비 1.5 ATR 다이내믹 트레일링
        if exit_event is None and pos["t2_hit"]:
            dynamic_trail = floor_to_tick(pos["peak_price"] - (1.5 * pos["atr"]))
            pos["current_stop"] = max(pos["current_stop"], dynamic_trail)

        # 8. T3 도달 (잔여 전량 - 지정가 슬리피지 차감)
        if exit_event is None and b_high >= pos["t3"]:
            pos["t3_hit"] = True
            exit_event = "TARGET_3R_FULL"
            exit_price = floor_to_tick(pos["t3"] * (1.0 - limit_slippage))
        # 9. Time Stop (시장가 슬리피지 차감)
        elif exit_event is None and h_days >= max_holding:
            exit_event = "TIME_STOP"
            exit_price = floor_to_tick(b_close * (1.0 - market_slippage))

    if exit_event is not None:
        rem_shares = pos["shares"]
        final_gain = rem_shares * exit_price * (1.0 - sell_fee)
        cash_gain_today += final_gain
        tot_inflow = pos["cash_inflow"] + final_gain
        net_pnl_krw = tot_inflow - pos["total_invested"]
        net_ret_pct = (net_pnl_krw / pos["total_invested"]) * 100.0 if pos["total_invested"] > 0 else 0.0

        trade_record = {
            "종목코드": pos.get("code", ""),
            "진입일": pos["entry_date"],
            "청산일": cur_date_str,
            "진입트랙": pos["track"],
            "진입가": int(e_price),
            "최종청산가": int(exit_price),
            "투자원금(원)": int(pos["total_invested"]),
            "실현손익(원)": int(net_pnl_krw),
            "순수익률(%)": round(net_ret_pct, 2),
            "청산사유": exit_event,
            "보유거래일": h_days,
            "T1_달성": pos["t1_hit"],
            "T2_달성": pos["t2_hit"],
            "T3_달성": pos["t3_hit"],
            "모호봉_발생": is_ambiguous
        }
        return None, cash_gain_today, trade_record

    return pos, cash_gain_today, None

# -------------------------------------------------------------
# 8. [단일 종목] 일별 MTM 실계좌 백테스터
# -------------------------------------------------------------
def run_daily_mtm_backtest(df_hist, df_dual_regime, historical_supply_series=None, max_holding=10, initial_capital=30000000, risk_pct=1.0, 
                           buy_cost=0.05, sell_cost=0.20, market_slippage=0.0010, limit_slippage=0.0005, execution_mode="CONSERVATIVE", min_score=70):
    if len(df_hist) < 120 or df_dual_regime.empty: return pd.DataFrame(), pd.DataFrame(), {}
    
    df_p = df_hist.copy()
    df_p.index = pd.to_datetime(df_p.index).normalize()
    df_reg = df_dual_regime.copy()
    df_reg.index = pd.to_datetime(df_reg.index).normalize()
    
    buy_fee = buy_cost / 100.0
    sell_fee = sell_cost / 100.0

    cash = float(initial_capital)
    active_pos = None
    pending_entry = None
    
    daily_records = []
    closed_trades = []

    for idx in range(60, len(df_p)):
        cur_date = df_p.index[idx]
        cur_date_str = cur_date.strftime("%Y-%m-%d")
        bar = df_p.iloc[idx]
        b_open, b_high, b_low, b_close = float(bar['Open']), float(bar['High']), float(bar['Low']), float(bar['Close'])

        # A. 3-Track 진입 실행
        if active_pos is None and pending_entry is not None:
            executed, e_price = check_3track_execution(
                pending_entry["track"], b_open, b_high, b_low,
                pending_entry["ma20_entry"], pending_entry["high_20"],
                slippage_rate=market_slippage
            )
            if executed and e_price > 0:
                plan = calculate_execution_levels(df_p.iloc[:idx], e_price)
                if not plan["is_excessive_stop"]:
                    clean_amt = get_clean_avg_amount(df_p.iloc[:idx], 20)
                    pos = calculate_position_sizing(cash, risk_pct, pending_entry["multiplier"], e_price, plan["risk_1r"], plan["atr14"], clean_amt)
                    shares = pos["shares"]
                    if shares > 0:
                        req_cash = shares * e_price * (1.0 + buy_fee)
                        if cash >= req_cash:
                            cash -= req_cash
                            active_pos = {
                                "entry_date": cur_date_str,
                                "entry_price": e_price,
                                "shares": shares,
                                "orig_shares": shares,
                                "current_stop": plan["stop_price"],
                                "t1": plan["target_1r"],
                                "t2": plan["target_2r"],
                                "t3": plan["target_3r"],
                                "t1_hit": False,
                                "t2_hit": False,
                                "t3_hit": False,
                                "holding_days": 0,
                                "cash_inflow": 0.0,
                                "total_invested": req_cash,
                                "peak_price": e_price,
                                "atr": plan["atr14"],
                                "track": pending_entry["track"]
                            }
            pending_entry = None

        # B. 공통 포지션 관리 & 동적 트레일링
        if active_pos is not None:
            ma5_val = float(df_p['Close'].iloc[idx-4:idx+1].mean())
            prev_low = float(df_p['Low'].iloc[idx-1])
            active_pos, cash_gain, trade_rec = process_intraday_position(
                active_pos, bar, ma5_val, prev_low, cur_date_str, max_holding=max_holding, sell_fee=sell_fee,
                market_slippage=market_slippage, limit_slippage=limit_slippage, execution_mode=execution_mode
            )
            cash += cash_gain
            if trade_rec is not None:
                closed_trades.append(trade_rec)

        # C. 일별 종가 MTM 시가평가 (Close-to-Close)
        holding_val = (active_pos["shares"] * b_close) if active_pos is not None else 0.0
        tot_equity = cash + holding_val
        daily_records.append({
            "Date": cur_date, "Cash": cash, "Stock_Value": holding_val,
            "Total_Equity": tot_equity, "In_Position": (active_pos is not None)
        })

        # D. 장 마감 후 익일 시그널 평가 (과거 수급 시계열 자동 반영)
        if active_pos is None and pending_entry is None:
            past_reg = df_reg.loc[df_reg.index <= cur_date]
            reg_code = past_reg.iloc[-1]['Regime'] if not past_reg.empty else "NEUTRAL"
            ks_sub = df_reg[['KS_Close']].rename(columns={'KS_Close': 'Close'}).loc[:cur_date]

            has_frgn = False
            if historical_supply_series is not None and not historical_supply_series.empty:
                if cur_date in historical_supply_series.index:
                    has_frgn = bool(historical_supply_series.loc[cur_date] > 0)
                else:
                    past_s = historical_supply_series.loc[historical_supply_series.index <= cur_date]
                    if not past_s.empty: has_frgn = bool(past_s.iloc[-1] > 0)

            dec = evaluate_full_execution_pipeline(df_p.iloc[:idx+1], reg_code, kospi_slice=ks_sub, has_frgn=has_frgn, min_score=min_score)
            if "BUY" in dec["action"]:
                pending_entry = {
                    "regime": reg_code,
                    "multiplier": dec["multiplier"],
                    "track": dec["scored"]["track"],
                    "high_20": dec["scored"]["high_20"],
                    "ma20_entry": dec["scored"]["ma20"]
                }

    daily_df = pd.DataFrame(daily_records).set_index("Date")
    trades_df = pd.DataFrame(closed_trades)

    if daily_df.empty: return daily_df, trades_df, {}

    daily_df['Peak'] = daily_df['Total_Equity'].cummax()
    daily_df['Drawdown_pct'] = ((daily_df['Total_Equity'] - daily_df['Peak']) / daily_df['Peak']) * 100.0
    daily_mdd = round(abs(daily_df['Drawdown_pct'].min()), 2)
    tot_ret_pct = round(((daily_df['Total_Equity'].iloc[-1] - initial_capital) / initial_capital) * 100.0, 2)

    if not trades_df.empty:
        wins_krw = trades_df[trades_df['실현손익(원)'] > 0]['실현손익(원)']
        losses_krw = trades_df[trades_df['실현손익(원)'] < 0]['실현손익(원)'].abs()
        profit_factor = round(wins_krw.sum() / losses_krw.sum(), 2) if losses_krw.sum() > 0 else 99.9
        expectancy_krw = int(trades_df['실현손익(원)'].mean())
        win_rate = round((len(wins_krw) / len(trades_df)) * 100, 1)
        t1_rate = round((trades_df['T1_달성'].sum() / len(trades_df)) * 100, 1)
        t2_rate = round((trades_df['T2_달성'].sum() / len(trades_df)) * 100, 1)
        ambiguous_count = int(trades_df['모호봉_발생'].sum()) if '모호봉_발생' in trades_df.columns else 0
    else:
        profit_factor, expectancy_krw, win_rate, t1_rate, t2_rate, ambiguous_count = 0.0, 0, 0.0, 0.0, 0.0, 0

    metrics = {
        "final_equity": int(daily_df['Total_Equity'].iloc[-1]),
        "total_return_pct": tot_ret_pct,
        "daily_mdd": daily_mdd,
        "profit_factor": profit_factor,
        "expectancy_krw": expectancy_krw,
        "win_rate": win_rate,
        "t1_rate": t1_rate,
        "t2_rate": t2_rate,
        "ambiguous_count": ambiguous_count,
        "total_trades": len(trades_df)
    }

    return daily_df, trades_df, metrics

# -------------------------------------------------------------
# 9. [포트폴리오 바스켓] 다종목 동시 자본 배분 시뮬레이터
# -------------------------------------------------------------
def run_portfolio_backtest(ticker_list, df_dual_regime, df_krx=None, max_positions=3, max_holding=10, initial_capital=50000000, risk_pct=1.0,
                           buy_cost=0.05, sell_cost=0.20, market_slippage=0.0010, limit_slippage=0.0005, execution_mode="CONSERVATIVE", min_score=70):
    s_date = (datetime.date.today() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
    data_dict = {}

    for item in ticker_list:
        clean_item = str(item).strip()
        if not clean_item: continue
        code = None
        if df_krx is not None and not df_krx.empty:
            code, _ = find_stock_code(clean_item, df_krx)
        if not code and clean_item.isdigit():
            code = clean_item.zfill(6)
        if not code: continue

        try:
            h = fdr.DataReader(code, s_date)
            if h is not None and len(h) >= 120:
                h.index = pd.to_datetime(h.index).normalize()
                data_dict[code] = h
        except Exception:
            continue

    if not data_dict or df_dual_regime.empty: return pd.DataFrame(), pd.DataFrame(), {}

    common_dates = sorted(list(set.intersection(*[set(df.index) for df in data_dict.values()])))
    common_dates = [d for d in common_dates if d >= pd.Timestamp(s_date)]
    if len(common_dates) < 60: return pd.DataFrame(), pd.DataFrame(), {}

    cash = float(initial_capital)
    positions = {}
    pending_signals = []
    daily_records = []
    closed_trades = []

    buy_fee = buy_cost / 100.0
    sell_fee = sell_cost / 100.0

    for cur_date in common_dates[60:]:
        cur_date_str = cur_date.strftime("%Y-%m-%d")

        # 1. 3-Track 진입 실행 (포트폴리오 전체 시가 총자산 기준 사이징)
        if pending_signals and len(positions) < max_positions:
            pending_signals.sort(key=lambda x: x["score"], reverse=True)
            for item in pending_signals:
                code = item["code"]
                if len(positions) >= max_positions: break
                if code in positions: continue

                df_code = data_dict[code].loc[:cur_date]
                bar_today = df_code.iloc[-1]
                b_open, b_high, b_low = float(bar_today['Open']), float(bar_today['High']), float(bar_today['Low'])

                executed, e_price = check_3track_execution(
                    item["track"], b_open, b_high, b_low, item["ma20_entry"], item["high_20"],
                    slippage_rate=market_slippage
                )
                if not executed or e_price <= 0: continue

                plan = calculate_execution_levels(df_code.iloc[:-1], e_price)
                if plan["is_excessive_stop"]: continue

                stk_val_at_open = sum(p["shares"] * float(data_dict[c].loc[cur_date, 'Open']) for c, p in positions.items())
                equity_at_open = cash + stk_val_at_open
                
                clean_amt = get_clean_avg_amount(df_code.iloc[:-1], 20)
                pos_calc = calculate_position_sizing(
                    equity_at_open, risk_pct, item["multiplier"], e_price, plan["risk_1r"], plan["atr14"], clean_amt
                )
                target_shares = pos_calc["shares"]

                max_cash_shares = int(cash // (e_price * (1.0 + buy_fee)))
                shares = min(target_shares, max_cash_shares)
                req_cash = shares * e_price * (1.0 + buy_fee)

                if shares > 0 and cash >= req_cash:
                    cash -= req_cash
                    positions[code] = {
                        "code": code,
                        "entry_date": cur_date_str,
                        "entry_price": e_price,
                        "shares": shares,
                        "orig_shares": shares,
                        "current_stop": plan["stop_price"],
                        "t1": plan["target_1r"],
                        "t2": plan["target_2r"],
                        "t3": plan["target_3r"],
                        "t1_hit": False,
                        "t2_hit": False,
                        "t3_hit": False,
                        "holding_days": 0,
                        "cash_inflow": 0.0,
                        "total_invested": req_cash,
                        "peak_price": e_price,
                        "atr": plan["atr14"],
                        "track": item["track"]
                    }
            pending_signals = []

        # 2. 공통 포지션 관리 & 동적 트레일링
        closed_codes = []
        for code, p in positions.items():
            df_code = data_dict[code].loc[:cur_date]
            bar = df_code.iloc[-1]
            ma5_val = float(df_code['Close'].iloc[-5:].mean())
            prev_low = float(df_code['Low'].iloc[-2])

            updated_p, cash_gain, trade_rec = process_intraday_position(
                p, bar, ma5_val, prev_low, cur_date_str, max_holding=max_holding, sell_fee=sell_fee,
                market_slippage=market_slippage, limit_slippage=limit_slippage, execution_mode=execution_mode
            )
            cash += cash_gain
            if trade_rec is not None:
                closed_trades.append(trade_rec)
                closed_codes.append(code)
            else:
                positions[code] = updated_p

        for c in closed_codes: del positions[c]

        # 3. MTM 일별 시가평가 (Close-to-Close)
        stk_val = sum(p["shares"] * float(data_dict[c].loc[cur_date, 'Close']) for c, p in positions.items())
        daily_records.append({
            "Date": cur_date, "Cash": cash, "Stock_Value": stk_val,
            "Total_Equity": cash + stk_val, "Num_Positions": len(positions)
        })

        # 4. 익일 진입 스캔
        if len(positions) < max_positions:
            past_reg = df_dual_regime.loc[df_dual_regime.index <= cur_date]
            reg_code = past_reg.iloc[-1]['Regime'] if not past_reg.empty else "NEUTRAL"
            ks_sub = df_dual_regime[['KS_Close']].rename(columns={'KS_Close': 'Close'}).loc[:cur_date]

            for code in data_dict.keys():
                if code not in positions:
                    df_sub = data_dict[code].loc[:cur_date]
                    dec = evaluate_full_execution_pipeline(df_sub, reg_code, kospi_slice=ks_sub, has_frgn=False, min_score=min_score)
                    if "BUY" in dec["action"]:
                        pending_signals.append({
                            "code": code,
                            "multiplier": dec["multiplier"],
                            "score": dec["scored"]["score"],
                            "track": dec["scored"]["track"],
                            "high_20": dec["scored"]["high_20"],
                            "ma20_entry": dec["scored"]["ma20"]
                        })

    port_df = pd.DataFrame(daily_records).set_index("Date")
    p_trades = pd.DataFrame(closed_trades)
    if port_df.empty: return port_df, p_trades, {}

    port_df['Peak'] = port_df['Total_Equity'].cummax()
    port_df['Drawdown_pct'] = ((port_df['Total_Equity'] - port_df['Peak']) / port_df['Peak']) * 100.0
    p_mdd = round(abs(port_df['Drawdown_pct'].min()), 2)
    p_ret = round(((port_df['Total_Equity'].iloc[-1] - initial_capital) / initial_capital) * 100.0, 2)

    if not p_trades.empty:
        wins_krw = p_trades[p_trades['실현손익(원)'] > 0]['실현손익(원)']
        losses_krw = p_trades[p_trades['실현손익(원)'] < 0]['실현손익(원)'].abs()
        p_pf = round(wins_krw.sum() / losses_krw.sum(), 2) if losses_krw.sum() > 0 else 99.9
        p_exp = int(p_trades['실현손익(원)'].mean())
        p_win = round((len(wins_krw) / len(p_trades)) * 100, 1)
        p_ambiguous = int(p_trades['모호봉_발생'].sum()) if '모호봉_발생' in p_trades.columns else 0
    else:
        p_pf, p_exp, p_win, p_ambiguous = 0.0, 0, 0.0, 0

    metrics = {
        "final_equity": int(port_df['Total_Equity'].iloc[-1]),
        "return_pct": p_ret,
        "daily_mdd": p_mdd,
        "profit_factor": p_pf,
        "expectancy_krw": p_exp,
        "win_rate": p_win,
        "ambiguous_count": p_ambiguous,
        "total_trades": len(p_trades)
    }

    return port_df, p_trades, metrics

# -------------------------------------------------------------
# 10. [Walk-Forward WFO & 3-모드 삼각비교 엔진]
# -------------------------------------------------------------
def run_tricorder_backtest(df_hist, df_dual_regime, historical_supply_series=None, initial_capital=30000000, risk_pct=1.0):
    """Conservative, Neutral, Optimistic 3개 모드를 동시 연산하여 안전 마진과 불확실성 델타 산출"""
    results = {}
    for mode in ["CONSERVATIVE", "NEUTRAL", "OPTIMISTIC"]:
        daily_df, trades_df, met = run_daily_mtm_backtest(
            df_hist, df_dual_regime, historical_supply_series=historical_supply_series,
            initial_capital=initial_capital, risk_pct=risk_pct, execution_mode=mode
        )
        results[mode] = met
        results[f"{mode}_daily"] = daily_df
        results[f"{mode}_trades"] = trades_df

    if not results["CONSERVATIVE"]: return pd.DataFrame(), results

    comp_df = pd.DataFrame([
        {
            "체결 모드": "보수적 (Worst-Case Stop)",
            "최종 잔고": f"{results['CONSERVATIVE']['final_equity']:,}원",
            "수익률": f"{results['CONSERVATIVE']['total_return_pct']:+}%",
            "일별 MTM MDD": f"-{results['CONSERVATIVE']['daily_mdd']}%",
            "Profit Factor": results['CONSERVATIVE']['profit_factor'],
            "승률": f"{results['CONSERVATIVE']['win_rate']}%",
            "모호봉 건수": f"{results['CONSERVATIVE']['ambiguous_count']}건"
        },
        {
            "체결 모드": "중립적 (Candle-Flow)",
            "최종 잔고": f"{results['NEUTRAL']['final_equity']:,}원",
            "수익률": f"{results['NEUTRAL']['total_return_pct']:+}%",
            "일별 MTM MDD": f"-{results['NEUTRAL']['daily_mdd']}%",
            "Profit Factor": results['NEUTRAL']['profit_factor'],
            "승률": f"{results['NEUTRAL']['win_rate']}%",
            "모호봉 건수": f"{results['NEUTRAL']['ambiguous_count']}건"
        },
        {
            "체결 모드": "낙관적 (Target-First)",
            "최종 잔고": f"{results['OPTIMISTIC']['final_equity']:,}원",
            "수익률": f"{results['OPTIMISTIC']['total_return_pct']:+}%",
            "일별 MTM MDD": f"-{results['OPTIMISTIC']['daily_mdd']}%",
            "Profit Factor": results['OPTIMISTIC']['profit_factor'],
            "승률": f"{results['OPTIMISTIC']['win_rate']}%",
            "모호봉 건수": f"{results['OPTIMISTIC']['ambiguous_count']}건"
        }
    ])
    return comp_df, results

def run_slippage_stress_test(df_hist, df_dual_regime, historical_supply_series=None, initial_capital=30000000, risk_pct=1.0):
    slippage_levels = [0.0005, 0.0010, 0.0020, 0.0030, 0.0050]
    stress_results = []
    for s_rate in slippage_levels:
        _, _, met = run_daily_mtm_backtest(
            df_hist, df_dual_regime, historical_supply_series=historical_supply_series,
            initial_capital=initial_capital, risk_pct=risk_pct,
            market_slippage=s_rate, limit_slippage=s_rate/2.0
        )
        if met:
            stress_results.append({
                "시장가 슬리피지": f"{s_rate*100:.2f}%",
                "지정가 슬리피지": f"{s_rate*50:.2f}%",
                "수익률(%)": met["total_return_pct"],
                "일별 MTM MDD(%)": met["daily_mdd"],
                "Profit Factor": met["profit_factor"],
                "거래당 기댓값(원)": met["expectancy_krw"],
                "승률(%)": met["win_rate"]
            })
    return pd.DataFrame(stress_results)

def run_true_wfo(df_hist, df_dual_regime, historical_supply_series=None, windows=4, initial_capital=30000000):
    """
    진성 Walk-Forward Optimization (WFO):
    IS에서 최적 스코어(65, 70, 75, 80점)를 탐색한 뒤, 파라미터를 OOS에 블라인드 동결 적용하여 WFE를 산출
    """
    df_p = df_hist.copy()
    total_len = len(df_p)
    if total_len < 250: return pd.DataFrame()

    segment_len = total_len // (windows + 1)
    wf_records = []
    score_candidates = [65, 70, 75, 80]

    for w in range(windows):
        train_start = w * segment_len
        train_end = train_start + (segment_len * 2)
        test_end = min(train_end + segment_len, total_len)
        if train_end >= total_len or train_end >= test_end: break

        df_train = df_p.iloc[train_start:train_end]
        df_test = df_p.iloc[train_end:test_end]

        # 1. In-Sample(IS) 최적 파라미터 탐색
        best_score = 70
        best_is_pf = -1.0
        best_is_ret = 0.0

        for sc in score_candidates:
            _, _, m_tr = run_daily_mtm_backtest(
                df_train, df_dual_regime, historical_supply_series=historical_supply_series,
                initial_capital=initial_capital, min_score=sc
            )
            if m_tr and m_tr["profit_factor"] > best_is_pf:
                best_is_pf = m_tr["profit_factor"]
                best_is_ret = m_tr["total_return_pct"]
                best_score = sc

        # 2. Out-of-Sample(OOS) 블라인드 동결 검증
        _, _, m_te = run_daily_mtm_backtest(
            df_test, df_dual_regime, historical_supply_series=historical_supply_series,
            initial_capital=initial_capital, min_score=best_score
        )

        if m_te:
            oos_ret = m_te["total_return_pct"]
            wfe = round((oos_ret / best_is_ret) * 100, 1) if best_is_ret > 0 else 0.0

            wf_records.append({
                "구간": f"Window {w+1}",
                "IS 구간": f"{df_train.index[0].strftime('%y.%m')} ~ {df_train.index[-1].strftime('%y.%m')}",
                "최적 파라미터": f"Score {best_score}점",
                "IS 수익률(%)": best_is_ret,
                "IS PF": best_is_pf,
                "OOS 블라인드 구간": f"{df_test.index[0].strftime('%y.%m')} ~ {df_test.index[-1].strftime('%y.%m')}",
                "OOS 수익률(%)": oos_ret,
                "OOS PF": m_te["profit_factor"],
                "WFE(%)": f"{wfe}%"
            })

    return pd.DataFrame(wf_records)

# -------------------------------------------------------------
# [UI 라우팅 및 뷰 구성]
# -------------------------------------------------------------
dual_regime = get_current_dual_regime()

st.title("⚖️ QUANT-EXECUTION ENGINE PRO MAX")
st.caption("“주가를 예측하지 않고, **틀렸을 때 손실을 제한하고 맞았을 때 분할 익절**하도록 통제하는 기계적 실행 시스템”")

c_rg1, c_rg2 = st.columns([1, 3])
with c_rg1:
    st.metric("실시간 듀얼 시장 국면", dual_regime["status"], f"배분 배수: {dual_regime['multiplier']}x")
with c_rg2:
    st.info(f"💡 **시장 대응 지침**: {dual_regime['msg']}")

if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = "삼성전자"

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 STEP 1 · 실행 중심 퀀트 진단 & 포지션 계산",
    "📡 STEP 2 · 코스피 Buy Zone 스크리너",
    "🧪 STEP 3 · 단일 종목 실계좌 MTM 백테스트",
    "💼 STEP 4 · 다종목 포트폴리오 시뮬레이터",
    "🔬 STEP 5 · Walk-Forward(WFO) & 스트레스 테스트",
    "📖 시스템 무결성 헌장"
])

# -------------------------------------------------------------
# TAB 1: 퀀트 진단 & 포지션 계산
# -------------------------------------------------------------
with tab1:
    st.subheader("정밀 실행 계획 & 리스크 기반 포지션 산출")
    
    col1, col2, col3 = st.columns([2, 1.5, 1])
    with col1:
        target_stock = st.text_input("종목명 또는 6자리 코드", value=st.session_state["selected_stock"])
    with col2:
        acc_cap = st.number_input("내 총 계좌 잔고 (원)", min_value=1000000, value=30000000, step=5000000)
    with col3:
        risk_pct_input = st.slider("1회 허용 손실률 (%)", 0.3, 2.0, 1.0, step=0.1)

    if st.button("실행 계획 정밀 진단", type="primary"):
        with st.spinner("호가 검증 및 100점 파이프라인 연산 중..."):
            df_krx = load_krx_listing()
            code, name = find_stock_code(target_stock, df_krx)
            
            if not code:
                st.error("종목을 찾을 수 없습니다.")
            else:
                s_date = (datetime.date.today() - datetime.timedelta(days=350)).strftime("%Y-%m-%d")
                df_p = fdr.DataReader(code, s_date)
                df_ks = fdr.DataReader('KS11', s_date)
                
                if df_p is None or len(df_p) < 40:
                    st.error("데이터가 불충분합니다.")
                else:
                    supply = get_supply_data(code)
                    verdict = evaluate_full_execution_pipeline(
                        df_p, dual_regime["code"], kospi_slice=df_ks, has_frgn=(supply["frgn_5"] > 0)
                    )
                    levels = verdict["levels"]
                    scored = verdict["scored"]
                    curr_price = int(df_p['Close'].iloc[-1])

                    clean_amt = get_clean_avg_amount(df_p, 20)
                    pos = calculate_position_sizing(
                        acc_cap, risk_pct_input, verdict["multiplier"],
                        curr_price, levels["risk_1r"], levels["atr14"], clean_amt
                    )

                    st.markdown("---")
                    if "BUY" in verdict["action"]:
                        st.success(f"### {verdict['badge']}\n**진입 전략 트랙**: `{scored['track']}` | **지침**: {verdict['reason']}")
                    elif verdict["action"] == "WAIT":
                        st.warning(f"### {verdict['badge']}\n**지침**: {verdict['reason']}")
                    else:
                        st.error(f"### {verdict['badge']}\n**지침**: {verdict['reason']}")

                    st.markdown("#### 💯 100점 만점 퀀트 평가 내역")
                    dt_cols = st.columns(6)
                    for i, (k, v) in enumerate(scored["details"].items()):
                        dt_cols[i].metric(k, v)

                    if levels["resistance_warning"]:
                        st.warning(f"⚠️ **저항선 주의**: {levels['resistance_warning']}")

                    st.markdown("### 🛑 기계적 가격선 및 손익비 구조")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("진입 기준가", f"{levels['entry_price']:,}원", f"Wilder ATR: {levels['atr14']:,}원")
                    m2.metric("동적 손절선 (보수적)", f"{levels['stop_price']:,}원", f"{levels['loss_pct']}%", delta_color="inverse")
                    m3.metric("1차 익절가 (30% 매도 & 본전스탑)", f"{levels['target_1r']:,}원", "+1.0R")
                    m4.metric("2차 익절가 (40% 매도)", f"{levels['target_2r']:,}원", "+2.0R")

                    st.markdown("### 💼 계좌 리스크 기반 포지션 및 2-Track 손실 시뮬레이션")
                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("최종 권장 매수 수량", f"{pos['shares']:,}주", f"계좌 비중: {pos['weight_pct']}%")
                    p2.metric("수량 제한 요인", pos["limiting_factor"], f"실질 리스크: {pos['real_risk_pct']}%")
                    p3.metric("정상 손절 시 손실액", f"-{pos['normal_risk_loss']:,}원", "1R 원칙 통제")
                    p4.metric("스트레스 갭다운(-1ATR)", f"-{pos['worst_gap_loss']:,}원", "오버나이트 급락 가정", delta_color="inverse")

                    if client:
                        prompt = f"""
                        당신은 엄격한 퀀트 트레이딩 데스크의 리스크 통제 총괄입니다.
                        다음 정량 데이터를 바탕으로 {name}({code})의 기계적 매매 지침을 브리핑하세요.
                        - 듀얼 시장 국면: {dual_regime['status']} (실행 배수 {verdict['multiplier']}x)
                        - 퀀트 평가: {scored['score']}점 ({scored['grade']}), 추천 트랙: {scored['track']}
                        - 진입 기준가: {curr_price:,}원 / 손절가: {levels['stop_price']:,}원 ({levels['loss_pct']}%)
                        - 1차 목표가: {levels['target_1r']:,}원 / 2차 목표가: {levels['target_2r']:,}원
                        - 권장 주문 수량: {pos['shares']}주 (제약요인: {pos['limiting_factor']}, 정상 손실 -{pos['normal_risk_loss']:,}원)
                        - 저항선 경고: {levels['resistance_warning']}
                        *과도한 낙관을 배제하고 자본 보존과 손익비 관점에서 3~4문장으로 서술하세요.*
                        """
                        with st.spinner("AI 리스크 브리핑 작성 중..."):
                            rep = generate_ai_briefing(client, prompt)
                            st.info(f"**AI 리스크 브리핑**:\n\n{rep}")

# -------------------------------------------------------------
# TAB 2: 코스피 Buy Zone 스크리너 (PyArrow 문자열 타입 에러 방어)
# -------------------------------------------------------------
with tab2:
    st.subheader("코스피 Buy Zone 지지 및 반등 트리거 탐색기")
    st.caption("동일한 100점 파이프라인을 전 종목 풀에 실시간 적용하여 70점 이상 유효 종목만 선별합니다.")

    c_s1, c_s2 = st.columns(2)
    with c_s1:
        min_cap = st.number_input("시가총액 하한 (억원)", 500, 20000, 2000, step=500)
    with c_s2:
        scan_pool = st.slider("스캔 풀 (거래대금 상위)", 100, 300, 150, step=50)

    if st.button("Buy Zone 종목 스캔 시작", type="primary"):
        with st.spinner("코스피 후보군 100점 평가 검증 중..."):
            df_k = fdr.StockListing('KOSPI')
            if df_k is None or df_k.empty:
                st.error("코스피 종목 로드 실패.")
            else:
                df_k['Code'] = df_k['Code'].astype(str).str.zfill(6)
                for num_col in ['Close', 'Volume', 'Amount', 'Marcap']:
                    if num_col in df_k.columns:
                        df_k[num_col] = pd.to_numeric(df_k[num_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
                if 'Amount' not in df_k.columns or (df_k['Amount'] == 0).all():
                    df_k['Amount'] = df_k['Close'] * df_k['Volume']
                
                pool = df_k[df_k['Marcap'] >= (min_cap * 100000000)].sort_values(by="Amount", ascending=False).head(scan_pool)
                screened = []
                p_bar = st.progress(0)
                s_date = (datetime.date.today() - datetime.timedelta(days=120)).strftime("%Y-%m-%d")
                df_ks = fdr.DataReader('KS11', s_date)

                for idx, (_, r) in enumerate(pool.iterrows()):
                    p_bar.progress((idx + 1) / len(pool))
                    code, name = str(r['Code']).zfill(6), r['Name']
                    try:
                        h = fdr.DataReader(code, s_date)
                        if h is None or len(h) < 30: continue
                        
                        dec = evaluate_full_execution_pipeline(h, dual_regime["code"], kospi_slice=df_ks, has_frgn=False)
                        if "BUY" in dec["action"]:
                            c_p = int(h['Close'].iloc[-1])
                            lvs = dec["levels"]
                            screened.append({
                                "코드": code, "종목명": name, "현재가": c_p,
                                "스코어": f"{dec['scored']['score']}점",
                                "트랙": dec['scored']['track'],
                                "최종판정": dec["badge"], "손절가": lvs["stop_price"],
                                "손절폭(%)": lvs["loss_pct"], "1차목표(1R)": lvs["target_1r"],
                                "2차목표(2R)": lvs["target_2r"], "거래대금(억)": int(r['Amount'] / 100000000)
                            })
                    except Exception:
                        continue
                p_bar.empty()
                df_res = pd.DataFrame(screened)
                st.session_state["screened_res"] = df_res
                if not df_res.empty:
                    st.success(f"조건을 만족하는 **{len(df_res)}개 종목**이 포착되었습니다.")
                else:
                    st.warning("현재 매수 요건을 충족하는 종목이 없습니다.")

    if "screened_res" in st.session_state and not st.session_state["screened_res"].empty:
        df_show = st.session_state["screened_res"]
        st.dataframe(df_show, width='stretch')
        
        sel = st.selectbox("진단 탭으로 전송할 종목 선택", df_show["종목명"].tolist())
        if st.button("선택 종목을 STEP 1으로 전송"):
            st.session_state["selected_stock"] = sel
            st.info(f"'{sel}' 선택 완료. **STEP 1** 탭으로 이동하세요.")

# -------------------------------------------------------------
# TAB 3: 단일 종목 실계좌 MTM 백테스트 & 3-모드 삼각비교
# -------------------------------------------------------------
with tab3:
    st.subheader("단일 종목 일별 시가평가(Daily MTM) 실계좌 백테스트")
    st.caption("3-Track 실제 체결, 부분익절 누적 현금흐름 완결 회계, 과거 5일 외국인 수급 복원 및 일별 종가 MTM MDD를 반영합니다.")

    bt_stock = st.text_input("백테스트 대상 종목명 또는 코드", value=st.session_state["selected_stock"])
    c_b1, c_b2, c_b3, c_b4 = st.columns(4)
    with c_b1:
        bt_init_cap = st.number_input("시뮬레이션 초기 자본금 (원)", value=30000000, step=5000000)
    with c_b2:
        bt_risk_pct = st.slider("1회 거래 리스크 비율 (%)", 0.5, 2.0, 1.0, step=0.1)
    with c_b3:
        bt_hold = st.slider("최대 허용 보유 기간 (Time Stop)", 5, 20, 10)
    with c_b4:
        exec_mode = st.selectbox("기본 체결 모드", ["CONSERVATIVE", "NEUTRAL", "OPTIMISTIC"])

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_single_bt = st.button("일별 MTM 백테스트 가동", type="primary")
    with col_btn2:
        start_tricorder_bt = st.button("⚡ 3-모드 동시 삼각비교(Tricorder) 가동")

    if start_single_bt or start_tricorder_bt:
        with st.spinner(f"'{bt_stock}' 과거 2년 수급 시계열 크롤링 및 MTM 연산 중..."):
            df_krx = load_krx_listing()
            b_code, b_name = find_stock_code(bt_stock, df_krx)
            if not b_code:
                st.error("종목을 찾을 수 없습니다.")
            else:
                s_date = (datetime.date.today() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
                df_bt = fdr.DataReader(b_code, s_date)
                dual_hist = load_historical_dual_regime(s_date)
                frgn_hist = load_historical_foreign_series(b_code)

                if start_tricorder_bt:
                    comp_df, tri_res = run_tricorder_backtest(
                        df_bt, dual_hist, historical_supply_series=frgn_hist,
                        initial_capital=bt_init_cap, risk_pct=bt_risk_pct
                    )
                    st.markdown("### ⚡ 3-모드 체결 불확실성 삼각비교 (Tricorder Comparison)")
                    st.dataframe(comp_df, width='stretch')
                    st.line_chart(pd.DataFrame({
                        "보수적 (Worst-Case)": tri_res["CONSERVATIVE_daily"]["Total_Equity"],
                        "중립적 (Candle-Flow)": tri_res["NEUTRAL_daily"]["Total_Equity"],
                        "낙관적 (Target-First)": tri_res["OPTIMISTIC_daily"]["Total_Equity"]
                    }))
                else:
                    daily_df, trades_df, met = run_daily_mtm_backtest(
                        df_bt, dual_hist, historical_supply_series=frgn_hist, max_holding=bt_hold,
                        initial_capital=bt_init_cap, risk_pct=bt_risk_pct,
                        execution_mode=exec_mode
                    )

                    if daily_df.empty:
                        st.warning("과거 2년간 모든 엄격한 필터를 통과한 거래가 없습니다.")
                    else:
                        st.markdown(f"### 📊 '{b_name}'({b_code}) 일별 MTM 실계좌 성과 요약")
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("최종 계좌 잔고", f"{met['final_equity']:,}원", f"{met['total_return_pct']:+}%")
                        k2.metric("일별 종가 MTM MDD", f"-{met['daily_mdd']}%", "Close-to-Close", delta_color="inverse")
                        k3.metric("Profit Factor (원화 기준)", f"{met['profit_factor']}")
                        k4.metric("거래당 기댓값 (Expectancy)", f"{met['expectancy_krw']:,}원")

                        k5, k6, k7, k8 = st.columns(4)
                        k5.metric("승률 (순익 > 0)", f"{met['win_rate']}%", f"총 {met['total_trades']}회")
                        k6.metric("T1 도달률 (+1.0R)", f"{met['t1_rate']}%")
                        k7.metric("T2 도달률 (+2.0R)", f"{met['t2_rate']}%")
                        k8.metric("모호봉(Ambiguous) 발생", f"{met['ambiguous_count']}건", f"적용모드: {exec_mode}")

                        st.line_chart(daily_df['Total_Equity'])
                        st.markdown("#### 📋 완료된 매매 내역 (원화 정밀 회계)")
                        st.dataframe(trades_df, width='stretch')

# -------------------------------------------------------------
# TAB 4: 다종목 포트폴리오 시뮬레이터
# -------------------------------------------------------------
with tab4:
    st.subheader("다종목 동시 보유 포트폴리오 시뮬레이터 (Portfolio MTM)")
    st.caption("전체 계좌 자산 기준 1% 리스크 사이징, 점수 순 슬롯 배정 및 3-Track과 동적 Trailing을 완전 일치 검증합니다.")

    default_tickers = "005930, 000660, 005380, 035420"
    basket_input = st.text_input("포트폴리오 바스켓 (종목명 또는 코드, 콤마 구분)", value=default_tickers)
    
    cp1, cp2, cp3, cp4 = st.columns(4)
    with cp1:
        port_capital = st.number_input("포트폴리오 초기 총자산 (원)", value=50000000, step=10000000)
    with cp2:
        max_slots = st.slider("동시 최대 보유 종목 수 (슬롯)", 2, 5, 3)
    with cp3:
        port_risk = st.slider("종목당 허용 리스크 (%)", 0.5, 1.5, 1.0, step=0.1, help="포트폴리오 전체 총자산 대비 1회 허용 손실 비율")
    with cp4:
        p_exec_mode = st.selectbox("포트폴리오 체결 모드", ["CONSERVATIVE", "NEUTRAL", "OPTIMISTIC"], key="port_exec_mode")

    if st.button("포트폴리오 백테스트 실행", type="primary"):
        with st.spinner("다종목 전체자산 기준 사이징 & 3-Track 동시 시뮬레이션 중..."):
            tickers = [t.strip() for t in basket_input.split(",") if t.strip()]
            df_krx = load_krx_listing()
            s_date = (datetime.date.today() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
            dual_hist = load_historical_dual_regime(s_date)
            
            p_df, p_trades, p_met = run_portfolio_backtest(
                tickers, dual_hist, df_krx=df_krx, max_positions=max_slots,
                initial_capital=port_capital, risk_pct=port_risk, execution_mode=p_exec_mode
            )

            if p_df.empty:
                st.warning("포트폴리오 진입 조건을 만족한 거래가 없습니다.")
            else:
                st.markdown("### 📊 다종목 포트폴리오 운용 성과")
                pk1, pk2, pk3, pk4 = st.columns(4)
                pk1.metric("포트폴리오 최종 잔고", f"{p_met['final_equity']:,}원", f"{p_met['return_pct']:+}%")
                pk2.metric("일별 종가 MTM MDD", f"-{p_met['daily_mdd']}%", "Close-to-Close", delta_color="inverse")
                pk3.metric("Profit Factor", f"{p_met['profit_factor']}")
                pk4.metric("거래당 기댓값", f"{p_met['expectancy_krw']:,}원")

                st.line_chart(p_df['Total_Equity'])
                st.markdown("#### 📋 포트폴리오 체결 내역 (동일 엔진 완결)")
                st.dataframe(p_trades, width='stretch')

# -------------------------------------------------------------
# TAB 5: Walk-Forward WFO & 슬리피지 스트레스 테스트
# -------------------------------------------------------------
with tab5:
    st.subheader("🔬 진성 Walk-Forward Optimization (WFO) & 슬리피지 부하 검증")
    st.caption("IS에서 파라미터를 탐색한 뒤 OOS에 블라인드 동결 적용하는 WFO 효율(WFE)과 슬리피지 급증 시 임계점을 평가합니다.")

    test_stock = st.text_input("검증 대상 종목명 또는 코드", value=st.session_state["selected_stock"], key="stress_stock")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("슬리피지 민감도 테스트 가동", type="primary"):
            with st.spinner("슬리피지 5단계 부하 시뮬레이션 중..."):
                df_krx = load_krx_listing()
                b_code, _ = find_stock_code(test_stock, df_krx)
                if b_code:
                    s_date = (datetime.date.today() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
                    df_bt = fdr.DataReader(b_code, s_date)
                    dual_hist = load_historical_dual_regime(s_date)
                    frgn_hist = load_historical_foreign_series(b_code)
                    stress_df = run_slippage_stress_test(df_bt, dual_hist, historical_supply_series=frgn_hist)
                    st.markdown("#### 📊 슬리피지 단계별 성과 변동표")
                    st.dataframe(stress_df, width='stretch')
                else:
                    st.error("종목을 찾을 수 없습니다.")

    with col_t2:
        if st.button("진성 Walk-Forward(WFO) 분석 가동", type="primary"):
            with st.spinner("4분할 윈도우 In-Sample 최적화 -> OOS 블라인드 평가 중..."):
                df_krx = load_krx_listing()
                b_code, _ = find_stock_code(test_stock, df_krx)
                if b_code:
                    s_date = (datetime.date.today() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
                    df_bt = fdr.DataReader(b_code, s_date)
                    dual_hist = load_historical_dual_regime(s_date)
                    frgn_hist = load_historical_foreign_series(b_code)
                    wf_df = run_true_wfo(df_bt, dual_hist, historical_supply_series=frgn_hist)
                    st.markdown("#### 📊 Walk-Forward Efficiency (WFE) 결과")
                    st.dataframe(wf_df, width='stretch')
                else:
                    st.error("종목을 찾을 수 없습니다.")

# -------------------------------------------------------------
# TAB 6: 시스템 무결성 헌장
# -------------------------------------------------------------
with tab6:
    st.subheader("📖 QUANT-EXECUTION 무결성 실행 원칙")
    st.markdown(r"""
    1. **단일/포트폴리오 엔진 완전 일치 (Unified Core Execution)**:
       * 단일 종목 백테스트와 다종목 포트폴리오 백테스트는 오직 `check_3track_execution`과 `process_intraday_position` 공통 함수만을 공유합니다.
       * 3-Track 진입 체결부터 +1.5R/+2.0R 이익보호, 1.5 ATR Trailing, Trend Break 청산까지 완전히 동일하게 작동합니다.
    2. **전체 자산 기준 리스크 사이징 (Portfolio Equity Risk Basis)**:
       * 포트폴리오 백테스트 시 슬롯 배분 현금이 아닌 현재 포트폴리오 전체 총평가자산($Cash + \sum Value_{open}$)을 기준으로 1% 리스크와 단일 종목 30% 비중 상한을 계산하여 현실적 자본 할당을 구현합니다.
    3. **전면 슬리피지 및 다단계 분할 회계 원칙 (Full Friction Accounting)**:
       * 최종 청산 가격으로 전체 수익률을 왜곡하지 않고, $T_1(30\%)$, $T_2(40\%)$, 잔여분의 실제 매도 유입 현금을 각각 집계하며, 지정가 익절(0.05%) 및 시장가 청산(0.10%) 슬리피지를 전면 차감하여 손익비를 계산합니다.
    4. **정합적 손절 통제 원칙 (Harmonized Risk Defense)**:
       * 손절선을 인위적인 범위(-2.5% ~ -7.5%)로 왜곡하지 않고 실제 구조/변동성 지지선에 정확히 배치하며, 리스크는 포지션 수량 조절을 최우선으로 하되 손절폭이 -8.5%를 초과하는 종목은 거래 자체를 거부(WAIT)하여 자본 회전율을 방어합니다.
    5. **일별 종가 MTM 평가 기준 (Close-to-Close MTM Transparency)**:
       * 백테스트의 MDD 및 자산 곡선은 거래 종료 시점 기준이 아닌 매일 장 마감 종가 기준의 시가평가(MTM)로 측정되며, 장중 틱 단위 미세 낙폭은 배제된 일봉 기준 지표임을 명확히 규정합니다.
    6. **모호봉 삼각비교 원칙 (Tricorder Comparison Policy)**:
       * 당일 고가와 저가가 목표가와 손절가를 동시에 건드리는 모호한 일봉 발생 시, 보수적 모드(CONSERVATIVE), 중립 모드(NEUTRAL), 낙관 모드(OPTIMISTIC)를 동시 연산하여 안전 마진과 불확실성 델타를 명시합니다.
    """)
