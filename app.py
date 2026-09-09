import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import io
import time
import math
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import FinanceDataReader as fdr
from google import genai

# =============================================================
# [API 및 모델 설정]
# =============================================================
DEFAULT_API_KEY = "여기에_AQ로_시작하는_키를_붙여넣으세요"
MODEL_CANDIDATES = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]

# -------------------------------------------------------------
# 0. UI 설정 및 API Key 로드 (키가 없어도 기본 기능 가동)
# -------------------------------------------------------------
st.set_page_config(
    page_title="종목 검색기 & 퀀트 실행 시스템 (QUANT - EXECUTION)",
    page_icon="📈",
    layout="wide"
)

secret_key = ""
try:
    if "GEMINI_API_KEY" in st.secrets:
        secret_key = st.secrets["GEMINI_API_KEY"].strip()
except Exception:
    pass

st.sidebar.header("🔑 Gemini API 설정")
input_key = st.sidebar.text_input(
    "API Key (자동 로드됨)",
    value=secret_key if secret_key else ("" if "여기에" in DEFAULT_API_KEY else DEFAULT_API_KEY),
    type="password",
    key="gemini_api_key_input",
    help="AI 분석 브리핑 기능을 이용할 때 필요합니다."
)

active_key = input_key.strip() if input_key.strip() else (secret_key if secret_key else (DEFAULT_API_KEY if "여기에" not in DEFAULT_API_KEY else ""))

client = None
if active_key:
    try:
        client = genai.Client(api_key=active_key)
        st.sidebar.success("✅ API 키 인증 완료")
    except Exception as e:
        st.sidebar.error(f"API 클라이언트 초기화 오류: {e}")
else:
    st.sidebar.info("💡 AI 브리핑 기능을 쓰시려면 API 키를 입력해 주세요. (기본 퀀트/스크리너는 키 없이 사용 가능)")

# -------------------------------------------------------------
# [보조 유틸] 국내 주식 호가 단위 세분화 (예외 방어 적용)
# -------------------------------------------------------------
def get_tick_size(price):
    try:
        if price is None or pd.isna(price): return 1
        p = abs(float(price))
    except (ValueError, TypeError):
        return 1
    if p < 2000: return 1
    elif p < 5000: return 5
    elif p < 20000: return 10
    elif p < 50000: return 50
    elif p < 200000: return 100
    elif p < 500000: return 500
    else: return 1000

def floor_to_tick(price):
    try:
        if price is None or pd.isna(price) or float(price) <= 0: return 0
        p = float(price)
        t = get_tick_size(p)
        return int(math.floor(p / t) * t)
    except (ValueError, TypeError):
        return 0

def ceil_to_tick(price):
    try:
        if price is None or pd.isna(price) or float(price) <= 0: return 0
        p = float(price)
        t = get_tick_size(p)
        return int(math.ceil(p / t) * t)
    except (ValueError, TypeError):
        return 0

def round_to_tick(price):
    try:
        if price is None or pd.isna(price) or float(price) <= 0: return 0
        p = float(price)
        t = get_tick_size(p)
        return int(round(p / t) * t)
    except (ValueError, TypeError):
        return 0

# -------------------------------------------------------------
# [안전 호출 함수] Gemini SDK
# -------------------------------------------------------------
def get_live_models(_client):
    preferred = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
    try:
        live_list = []
        for m in _client.models.list():
            actions = getattr(m, 'supported_actions', []) or []
            if 'generateContent' in actions:
                clean_name = m.name.replace('models/', '')
                live_list.append(clean_name)
        matched = [p for p in preferred if p in live_list]
        for l in live_list:
            if l not in matched and "flash" in l:
                matched.append(l)
        if matched:
            return matched
    except Exception:
        pass
    return MODEL_CANDIDATES

def generate_content_with_retry(client, prompt, max_retries=3):
    last_error = None
    target_models = get_live_models(client)
    retry_codes = ["429", "500", "502", "503", "504", "UNAVAILABLE"]
    for model_name in target_models:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if response and response.text:
                    return response.text
                return "⚠️ AI 응답 결과가 비어 있습니다."
            except Exception as e:
                err_str = str(e)
                last_error = err_str
                if any(code in err_str for code in retry_codes):
                    time.sleep((attempt + 1) * 2)
                    continue
                else:
                    break
    raise Exception(f"AI 호출 실패: {last_error}")

# -------------------------------------------------------------
# [전종목 마스터 로더] 한국거래소 KIND 공식 상장법인
# -------------------------------------------------------------
@st.cache_data(ttl=86400)
def load_all_krx_master():
    try:
        url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        res = requests.get(url, headers=headers, timeout=6)
        if res.status_code == 200:
            dfs = pd.read_html(io.BytesIO(res.content), encoding="cp949")
            if dfs and not dfs[0].empty:
                df = dfs[0][['회사명', '종목코드']].copy()
                df.columns = ['Name', 'Code']
                df['Code'] = df['Code'].astype(str).str.zfill(6)
                df = df.drop_duplicates(subset=['Code']).drop_duplicates(subset=['Name']).reset_index(drop=True)
                return df
    except Exception:
        pass
    return pd.DataFrame()

# -------------------------------------------------------------
# [데이터 로더] KRX 상장주식 (P0/P1 버그 해결: 동적 헤더 매핑 및 시세 검증)
# -------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_krx_listing():
    # 1. FinanceDataReader 시도 (시세 필수 열 엄격 검증)
    try:
        df = fdr.StockListing('KRX')
        if df is not None and not df.empty and 'Code' in df.columns and 'Name' in df.columns:
            req_market_cols = ['Close', 'Volume', 'Marcap']
            # 필수 열이 존재하고, 전량 0 또는 NaN이 아닌 유효 데이터인지 확인
            if all(c in df.columns for c in req_market_cols):
                has_valid_close = not (df['Close'].fillna(0) <= 0).all()
                has_valid_marcap = not (df['Marcap'].fillna(0) <= 0).all()
                if has_valid_close and has_valid_marcap:
                    df['Code'] = df['Code'].astype(str).str.zfill(6)
                    if 'Amount' not in df.columns or df['Amount'].isnull().all() or (df['Amount'].fillna(0) == 0).all():
                        df['Amount'] = df['Close'].fillna(0) * df['Volume'].fillna(0)
                    else:
                        df['Amount'] = df['Amount'].fillna(df['Close'].fillna(0) * df['Volume'].fillna(0))
                    
                    df = df.drop_duplicates(subset=['Code']).reset_index(drop=True)
                    df['DataSource'] = 'FinanceDataReader'
                    return df
    except Exception:
        pass

    # 2. 네이버 시가총액 백업 크롤러 (동적 헤더 인덱스 매핑)
    try:
        items = []
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        for sosok in [0, 1]:  # 0: 코스피, 1: 코스닥
            for page in range(1, 35):
                url = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}"
                res = requests.get(url, headers=headers, timeout=5)
                if res.status_code != 200:
                    break
                soup = BeautifulSoup(res.content.decode('cp949', errors='ignore'), 'html.parser')
                table = soup.select_one('table.type_2')
                if not table:
                    break
                
                # thead에서 열 위치 동적 파악
                th_elements = table.select('thead th')
                col_map = {th.text.strip(): i for i, th in enumerate(th_elements)}
                
                price_idx = col_map.get('현재가')
                marcap_idx = col_map.get('시가총액')
                vol_idx = col_map.get('거래량')
                
                if price_idx is None or marcap_idx is None or vol_idx is None:
                    break

                rows = table.select('tbody tr')
                found_valid = False
                for tr in rows:
                    a_tag = tr.select_one('a.tltle')
                    if not a_tag:
                        continue
                    name = a_tag.text.strip()
                    href = a_tag.get('href', '')
                    if 'code=' not in href:
                        continue
                    code = href.split('code=')[-1].split('&')[0].strip().zfill(6)
                    
                    tds = tr.select('td')
                    max_idx = max(price_idx, marcap_idx, vol_idx)
                    if len(tds) <= max_idx:
                        continue
                    try:
                        price_val = float(tds[price_idx].text.strip().replace(',', ''))
                        marcap_val = float(tds[marcap_idx].text.strip().replace(',', '')) * 100_000_000
                        vol_val = float(tds[vol_idx].text.strip().replace(',', ''))
                        
                        if price_val > 0 and marcap_val > 0:
                            items.append({
                                'Code': code,
                                'Name': name,
                                'Close': price_val,
                                'Marcap': marcap_val,
                                'Volume': vol_val,
                                'Amount': price_val * vol_val
                            })
                            found_valid = True
                    except Exception:
                        continue
                if not found_valid:
                    break

        if items:
            df_res = pd.DataFrame(items)
            df_res = df_res.drop_duplicates(subset=['Code']).reset_index(drop=True)
            df_res['DataSource'] = '네이버 금융 백업'
            return df_res
    except Exception:
        pass

    return pd.DataFrame()

# -------------------------------------------------------------
# [스크리너 성능] 최근 시세 지표를 제한 병렬 조회하고 1시간 캐시
# -------------------------------------------------------------
def _get_screening_metrics(code, start_date):
    """Streamlit UI와 분리된 단일 종목 조회 함수."""
    try:
        hist = fdr.DataReader(code, start_date)
        required_cols = ['Close', 'Volume']
        if hist is None or len(hist) < 25 or not all(c in hist.columns for c in required_cols):
            return {"code": code, "ok": False, "error": "최근 OHLCV 데이터 부족"}

        last_close = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        close_5d = hist['Close'].iloc[-6]
        last_vol = hist['Volume'].iloc[-1]
        prev_vol = hist['Volume'].iloc[-2]
        avg_vol_20d = hist['Volume'].iloc[-21:-1].mean()

        values = [last_close, prev_close, close_5d, last_vol, prev_vol, avg_vol_20d]
        if any(pd.isna(v) for v in values) or prev_close <= 0 or close_5d <= 0 or last_vol < 0:
            return {"code": code, "ok": False, "error": "가격 또는 거래량 결측/비정상"}

        return {
            "code": code,
            "ok": True,
            "last_close": float(last_close),
            "return_1d": (last_close - prev_close) / prev_close * 100,
            "return_5d": (last_close - close_5d) / close_5d * 100,
            "vol_ratio_20d": last_vol / max(avg_vol_20d, 1),
            "vol_ratio_prev": last_vol / max(prev_vol, 1),
        }
    except Exception as e:
        return {"code": code, "ok": False, "error": str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def load_screening_metrics_batch(codes, scan_date):
    """같은 날짜·대상 풀은 재실행 시 네트워크 호출 없이 재사용한다."""
    start_date = (datetime.datetime.fromisoformat(scan_date) - datetime.timedelta(days=60)).strftime("%Y-%m-%d")
    workers = min(8, len(codes))
    if workers == 0:
        return []

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_get_screening_metrics, code, start_date) for code in codes]
        for future in as_completed(futures):
            results.append(future.result())
    return results

# -------------------------------------------------------------
# [안전 종목 검색 함수]
# -------------------------------------------------------------
def find_stock_code(target_stock, df_krx):
    query = target_stock.strip()
    if not query:
        return None, None

    # 1. 6자리 종목코드 직접 입력
    if query.isdigit() and len(query) == 6:
        code = query
        name = query
        master = load_all_krx_master()
        if not master.empty:
            m = master[master['Code'] == code]
            if not m.empty:
                name = m.iloc[0]['Name']
        return code, name

    # 2. KIND 전종목 마스터 매칭
    query_clean = query.upper().replace(" ", "")
    master = load_all_krx_master()
    if not master.empty:
        names_clean = master['Name'].astype(str).str.upper().str.replace(" ", "")
        matched = master[names_clean == query_clean]
        if not matched.empty:
            return matched.iloc[0]['Code'], matched.iloc[0]['Name']

    # 3. df_krx 내 대조
    if df_krx is not None and not df_krx.empty and 'Name' in df_krx.columns:
        names_clean = df_krx['Name'].astype(str).str.upper().str.replace(" ", "")
        matched = df_krx[names_clean == query_clean]
        if not matched.empty:
            return matched.iloc[0]['Code'], matched.iloc[0]['Name']

    # 4. 네이버 공식 자동완성 백업
    try:
        url_ac = f"https://ac.finance.naver.com/ac?q={urllib.parse.quote(query)}&st=1&r_format=json&r_enc=utf-8"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://finance.naver.com/"
        }
        r = requests.get(url_ac, headers=headers, timeout=3)
        if r.status_code == 200:
            data = r.json()
            items = data.get('items', [])
            if items and len(items[0]) > 0:
                first = items[0][0]
                return str(first[0]).strip().zfill(6), str(first[1]).strip()
    except Exception:
        pass

    return None, None

# -------------------------------------------------------------
# [보조 함수] 네이버 금융 세부 펀더멘털 & 일별 수급 수집
# -------------------------------------------------------------
def get_comprehensive_stock_data(code):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    data = {
        "per": None, "pbr": None, "frgn_5": 0, "frgn_10": 0, "frgn_20": 0,
        "inst_5": 0, "inst_10": 0, "inst_20": 0, "data_confidence": "B (수급 미확인)"
    }
    
    # 1. 수급 데이터 수집
    try:
        url_sise = f"https://finance.naver.com/item/frgn.naver?code={code}"
        res = requests.get(url_sise, headers=headers, timeout=5)
        if res.status_code == 200:
            dfs = pd.read_html(io.BytesIO(res.content), encoding="cp949")
            df_frgn = None
            for t in dfs:
                cols_str = str(t.columns)
                if "기관" in cols_str and "외국인" in cols_str:
                    df_frgn = t
                    break
            if df_frgn is not None:
                df_frgn.columns = ['_'.join(col).strip() if isinstance(col, tuple) else str(col) for col in df_frgn.columns]
                df_frgn = df_frgn.dropna(subset=[df_frgn.columns[0]])
                
                inst_cols = [c for c in df_frgn.columns if "기관" in c and "순매" in c]
                frgn_cols = [c for c in df_frgn.columns if "외국인" in c and "순매" in c]
                
                if inst_cols and frgn_cols:
                    inst_vals = pd.to_numeric(df_frgn[inst_cols[0]].astype(str).str.replace(',', ''), errors='coerce').fillna(0).tolist()
                    frgn_vals = pd.to_numeric(df_frgn[frgn_cols[0]].astype(str).str.replace(',', ''), errors='coerce').fillna(0).tolist()
                    
                    data["frgn_5"] = int(sum(frgn_vals[:5]))
                    data["frgn_10"] = int(sum(frgn_vals[:10]))
                    data["frgn_20"] = int(sum(frgn_vals[:20]))
                    data["inst_5"] = int(sum(inst_vals[:5]))
                    data["inst_10"] = int(sum(inst_vals[:10]))
                    data["inst_20"] = int(sum(inst_vals[:20]))
                    data["data_confidence"] = "A"
    except Exception:
        data["data_confidence"] = "B (수급 미확인)"

    # 2. 재무 밸류에이션 수집
    try:
        url_main = f"https://finance.naver.com/item/main.naver?code={code}"
        res_main = requests.get(url_main, headers=headers, timeout=5)
        if res_main.status_code == 200:
            soup = BeautifulSoup(res_main.content.decode('cp949', errors='ignore'), 'html.parser')
            per_tag = soup.select_one('#_per')
            pbr_tag = soup.select_one('#_pbr')
            if per_tag:
                t_val = per_tag.text.strip().replace(',', '')
                try: data["per"] = float(t_val)
                except ValueError: data["per"] = None
            if pbr_tag:
                b_val = pbr_tag.text.strip().replace(',', '')
                try: data["pbr"] = float(b_val)
                except ValueError: data["pbr"] = None
    except Exception:
        pass

    return data

# -------------------------------------------------------------
# [보조 함수] Wilder's Smoothing RSI 및 표준 구간 해석기
# -------------------------------------------------------------
def calculate_wilder_rsi(series, period=14):
    if len(series) < period + 2:
        return 50.0
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean().iloc[-1]
    
    if pd.isna(avg_gain) or pd.isna(avg_loss) or (avg_gain == 0 and avg_loss == 0):
        return 50.0
    if avg_loss == 0:
        return 100.0
    if avg_gain == 0:
        return 0.0
        
    rs = avg_gain / avg_loss
    return round(float(100.0 - (100.0 / (1.0 + rs))), 1)

def get_rsi_status(rsi):
    if pd.isna(rsi):
        return "산출 불가"
    if rsi < 30:
        return "과매도"
    elif rsi < 50:
        return "약세 / 중립"
    elif rsi < 60:
        return "상승 모멘텀"
    elif rsi < 70:
        return "강세 / 과열 경계"
    elif rsi < 80:
        return "과열"
    else:
        return "극단적 과열"

# -------------------------------------------------------------
# [QUAD-MATRIX 퀀트 실행 엔진]
# -------------------------------------------------------------
def run_quad_execution_engine(df_hist, extra_data):
    curr_price = int(df_hist['Close'].iloc[-1])
    prev_close = int(df_hist['Close'].iloc[-2])
    today_open = int(df_hist['Open'].iloc[-1])
    today_low = int(df_hist['Low'].iloc[-1])
    
    ma5 = df_hist['Close'].rolling(5).mean().iloc[-1]
    ma20 = df_hist['Close'].rolling(20).mean().iloc[-1]
    ma60 = df_hist['Close'].rolling(60).mean().iloc[-1] if len(df_hist) >= 60 else np.nan
    ma120 = df_hist['Close'].rolling(120).mean().iloc[-1] if len(df_hist) >= 120 else np.nan
    
    # 20일 전고점: 직전 20거래일 (당일 봉 제외)
    high_20 = int(df_hist['High'].iloc[-21:-1].max())
    low_20 = int(df_hist['Low'].iloc[-21:-1].min())

    high_52w = int(df_hist['High'].max())
    if high_52w <= 0: high_52w = curr_price
    
    # Wilder RSI
    rsi_14 = calculate_wilder_rsi(df_hist['Close'], period=14)
    rsi_desc = get_rsi_status(rsi_14)
    
    # 이격도 vs 이격률
    if pd.notna(ma20) and ma20 > 0:
        disparity_ratio = round((curr_price / ma20) * 100, 1)
        disparity_pct = round(((curr_price - ma20) / ma20) * 100, 1)
        disp_display_str = f"{disparity_ratio}% ({'+' if disparity_pct > 0 else ''}{disparity_pct}%)"
    else:
        disparity_ratio = 100.0
        disparity_pct = 0.0
        disp_display_str = "100.0% (+0.0%)"
    
    vol_last = df_hist['Volume'].iloc[-1]
    avg_vol_20d = df_hist['Volume'].iloc[-21:-1].mean()
    if pd.isna(avg_vol_20d) or avg_vol_20d <= 0: avg_vol_20d = 1
    vol_ratio_20d = round(float(vol_last / avg_vol_20d), 2)

    # =========================================================
    # ① QUALITY (0~100점 정규화)
    # =========================================================
    s_val = 10
    per, pbr = extra_data.get('per'), extra_data.get('pbr')
    if per is not None:
        if 0 < per <= 15: s_val += 6
        elif 15 < per <= 30: s_val += 3
        elif per > 50: s_val -= 4
    if pbr is not None:
        if 0 < pbr <= 1.5: s_val += 4
        elif pbr > 4.0: s_val -= 2
    score_val = max(min(s_val, 20), 2)

    if extra_data['data_confidence'] == "A":
        f20, f5 = extra_data['frgn_20'], extra_data['frgn_5']
        i20, i5 = extra_data['inst_20'], extra_data['inst_5']
        s_frgn = 7 + (4 if f20 > 0 else 0) + (3 if f5 > 0 else (-2 if f20 > 0 else 0))
        s_inst = 5 + (3 if i20 > 0 else 0) + (2 if i5 > 0 else 0)
        score_supply = min(s_frgn + s_inst, 25)
    else:
        score_supply = 10

    if pd.notna(ma60):
        s_trend = 15 if curr_price >= ma5 >= ma20 >= ma60 else (10 if curr_price >= ma20 >= ma60 else 5)
    else:
        s_trend = 12 if curr_price >= ma5 >= ma20 else 6
    drop_from_52w = (high_52w - curr_price) / high_52w * 100
    s_high = 10 if drop_from_52w <= 5 else (7 if drop_from_52w <= 15 else (4 if drop_from_52w <= 30 else 2))
    score_trend = min(s_trend + s_high, 25)

    score_mom = 12 if vol_ratio_20d >= 2.0 else (8 if vol_ratio_20d >= 1.2 else 4)
    if curr_price > prev_close: score_mom += 8

    score_margin = 10 if abs(disparity_pct) <= 3.0 else (6 if abs(disparity_pct) <= 6.0 else 2)

    quality_score = score_val + score_supply + score_trend + score_mom + score_margin

    # =========================================================
    # ② TIMING (0~100점)
    # =========================================================
    pullback_min = floor_to_tick(ma20 * 0.99)
    pullback_max = ceil_to_tick(ma20 * 1.025)
    entry_ref_price = round_to_tick((pullback_min + pullback_max) / 2)

    timing_val = 50
    if 99.0 <= disparity_ratio <= 102.5:
        timing_val += 35
    elif 102.5 < disparity_ratio <= 105.0:
        timing_val += 15
    elif disparity_pct > 7.0:
        timing_val -= 35
    elif disparity_ratio < 95.0:
        timing_val -= 20

    dist_to_resist = (high_20 - curr_price) / curr_price * 100 if curr_price > 0 else 0
    if 0 < dist_to_resist <= 3.0:
        timing_val -= 15

    timing_score = max(min(timing_val, 100), 0)

    # =========================================================
    # ③ RISK (0~100점)
    # =========================================================
    risk_val = 10
    if rsi_14 >= 80: risk_val += 40
    elif rsi_14 >= 70: risk_val += 25
    elif rsi_14 >= 60: risk_val += 10
    
    if disparity_ratio >= 115: risk_val += 35
    elif disparity_ratio >= 110: risk_val += 20
    elif disparity_ratio >= 107: risk_val += 10
    
    if 0 < dist_to_resist <= 2.5:
        risk_val += 15
    if extra_data['data_confidence'] == "A" and extra_data['frgn_5'] < 0 and extra_data['inst_5'] < 0:
        risk_val += 10
    
    risk_score = max(min(risk_val, 100), 0)
    if risk_score <= 25:
        risk_label = "🟢 낮음 (안전)"
    elif risk_score <= 50:
        risk_label = "🟡 보통 (관리 가능)"
    elif risk_score <= 75:
        risk_label = "🟠 높음 (과열 경계)"
    else:
        risk_label = "🔴 극단적 과열 (진입 금지)"

    # =========================================================
    # ④ TRIGGER (0~100점)
    # =========================================================
    in_buy_zone = (pullback_min <= curr_price <= pullback_max)
    breakout_price = ceil_to_tick(high_20 * 1.005)
    is_breakout = (curr_price >= breakout_price and vol_ratio_20d >= 1.5 and curr_price > prev_close)

    trig_score = 0
    if is_breakout:
        trig_score = 85
        trigger_state = "🟢 ACTIVE (돌파 진입)"
        trigger_badge = "ACTIVE"
        trigger_action = "20일 전고점 돌파 + 거래량 충족 확인 (돌파 추종 유효)"
    elif in_buy_zone:
        trig_score = 40
        cond_count = 0
        if today_low >= int(ma20 * 0.985) and curr_price >= ma20:
            trig_score += 15; cond_count += 1
        if curr_price > today_open:
            trig_score += 15; cond_count += 1
        if today_low < prev_close and curr_price > prev_close:
            trig_score += 10; cond_count += 1
        if curr_price >= ma5:
            trig_score += 10; cond_count += 1
        if vol_ratio_20d >= 1.2 or (extra_data['data_confidence'] == "A" and extra_data['frgn_5'] > 0):
            trig_score += 10; cond_count += 1

        if cond_count >= 2 or trig_score >= 70:
            trigger_state = "🟢 ACTIVE (분할 진입)"
            trigger_badge = "ACTIVE"
            trigger_action = f"Buy Zone 내 반등 조건 {cond_count}개 충족 (기계적 분할 진입 가동)"
        elif cond_count == 1 or trig_score >= 50:
            trigger_state = "🟡 WATCH (조건 확인)"
            trigger_badge = "WATCH"
            trigger_action = f"Buy Zone 내 지지 확인 중 (조건 1개 충족, 추가 반등 신호 대기)"
        else:
            trigger_state = "🟠 WAIT (대기)"
            trigger_badge = "WAIT"
            trigger_action = "Buy Zone 진입했으나 반등 미확인 (떨어지는 칼날 매수 금지)"
    else:
        if disparity_pct > 7.0:
            trig_score = 20
            trigger_state = "🔴 OFF (진입 금지)"
            trigger_badge = "OFF"
            trigger_action = f"20일선 대비 +{disparity_pct:.1f}% 이격 과대 (추격매수 금지 / 눌림목 대기)"
        else:
            trig_score = 45
            trigger_state = "🟠 WAIT (대기)"
            trigger_badge = "WAIT"
            trigger_action = "Buy Zone 또는 돌파 기준선 미도달 (시나리오 관망)"

    # ⑤ 시장 상태 판정
    if pd.notna(ma60) and curr_price >= ma5 >= ma20 >= ma60:
        if disparity_pct > 7.0:
            market_state = "상승 추세 진행 / 현재 추격매수 불리 (눌림목 대기)"
            state_color = "warning"
        elif in_buy_zone:
            market_state = "상승 추세 내 최적 눌림목 구간 (반등 트리거 관찰)"
            state_color = "success"
        else:
            market_state = "상승 추세 유지 (보유선 트레일링 홀딩)"
            state_color = "info"
    elif pd.notna(ma60) and curr_price < ma20 and curr_price >= ma60:
        market_state = "단기 조정 진행 중 (60일선 지지력 테스트)"
        state_color = "info"
    else:
        market_state = "추세 훼손 또는 하락/비정배열 구간 (진입 보류)"
        state_color = "error"

    # ⑥ 손절선 설계 (호가 단위 내림 적용)
    target_entry = entry_ref_price if not in_buy_zone else curr_price
    stop_short_init = floor_to_tick(ma20 * 0.985)
    stop_swing_init = floor_to_tick(ma20 * 0.970)
    stop_mid_init = floor_to_tick(ma60 * 0.970) if pd.notna(ma60) else floor_to_tick(ma20 * 0.950)

    loss_short_pct = round((stop_short_init - target_entry) / target_entry * 100, 1)
    loss_swing_pct = round((stop_swing_init - target_entry) / target_entry * 100, 1)
    loss_mid_pct = round((stop_mid_init - target_entry) / target_entry * 100, 1)

    # ⑦ R-Multiple 목표 (손절폭 역전 검증 적용)
    if target_entry > stop_swing_init and stop_swing_init > 0:
        risk_unit_R = target_entry - stop_swing_init
        target_1R_val = ceil_to_tick(target_entry + risk_unit_R)
        target_2R_val = ceil_to_tick(target_entry + 2 * risk_unit_R)
        target_3R_val = ceil_to_tick(target_entry + 3 * risk_unit_R)

        r1_pct = round((target_1R_val - target_entry) / target_entry * 100, 1)
        r2_pct = round((target_2R_val - target_entry) / target_entry * 100, 1)
        r3_pct = round((target_3R_val - target_entry) / target_entry * 100, 1)

        target_1R_str = f"{target_1R_val:,}원 (+{r1_pct}%)"
        target_2R_str = f"{target_2R_val:,}원 (+{r2_pct}%)"
        target_3R_str = f"{target_3R_val:,}원 (+{r3_pct}%)"
    else:
        risk_unit_R = 0
        target_1R_str = "산출 불가 (손절폭 비정상)"
        target_2R_str = "산출 불가"
        target_3R_str = "산출 불가"

    return {
        "quality_score": quality_score,
        "timing_score": timing_score,
        "risk_score": risk_score,
        "risk_label": risk_label,
        "trig_score": trig_score,
        "trigger_state": trigger_state,
        "trigger_badge": trigger_badge,
        "trigger_action": trigger_action,
        "market_state": market_state,
        "state_color": state_color,
        "curr_price": curr_price,
        "target_entry": target_entry,
        "pullback_min": pullback_min,
        "pullback_max": pullback_max,
        "pullback_range": f"{pullback_min:,}원 ~ {pullback_max:,}원",
        "high_20": high_20,
        "breakout_price": breakout_price,
        "high_52w": high_52w,
        "ma5": int(ma5), "ma20": int(ma20),
        "ma60": int(ma60) if pd.notna(ma60) else None,
        "disp_display_str": disp_display_str,
        "disparity_pct": disparity_pct,
        "rsi_14": rsi_14,
        "rsi_desc": rsi_desc,
        "vol_ratio_20d": vol_ratio_20d,
        "stop_short_init": f"{stop_short_init:,}원 ({loss_short_pct}%)",
        "stop_swing_init": f"{stop_swing_init:,}원 ({loss_swing_pct}%)",
        "stop_mid_init": f"{stop_mid_init:,}원 ({loss_mid_pct}%)" if pd.notna(ma60) else "데이터 부족",
        "risk_unit_R": risk_unit_R,
        "target_1R": target_1R_str,
        "target_2R": target_2R_str,
        "target_3R": target_3R_str
    }

# -------------------------------------------------------------
# UI 대시보드
# -------------------------------------------------------------
st.title("📈 종목 검색기 & 퀀트 실행 시스템 (QUANT - EXECUTION)")
st.caption("ARCHITECTURE: 4대 독립 스코어링 (Quad-Matrix) + R-Multiple 익절 + 구조적 손절선")

tab1, tab2, tab3 = st.tabs(["STEP 1 · 주도 후보 스크리너", "STEP 2 · 퀀트 실행 분석 (Quad-Matrix)", "📖 시스템 실행 매뉴얼"])

# -------------------------------------------------------------
# STEP 1: 수급 테마 검색
# -------------------------------------------------------------
with tab1:
    st.subheader("🎯 STEP 1 · 주도 수급 스크리너")
    st.caption("주도주 조건: 5일 누적 상승률 ≥ 7% | 거래대금 상위 설정 비율(상세 조회 상한 적용) | 20일 거래량 ≥ 2.0x | 시총 ≥ 2,000억 | 당일 등락률 ≥ -2.0%")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        min_return_5d = st.slider("5일 누적 상승률 하한 (%)", 0, 30, 7)
        top_amount_pct = st.slider("거래대금 상위 비율 (%)", 5, 50, 20, step=5)
    with col2:
        vol_mult_20d = st.slider("20일 평균 거래량 대비 배수", 1.0, 5.0, 2.0, step=0.1)
        vol_mult_prev = st.slider("전일 거래량 대비 배수", 1.0, 3.0, 1.5, step=0.1)
    with col3:
        min_market_cap = st.number_input("시총 하한 (억원)", 500, 10000, 2000, step=500)
        min_daily_return = st.slider("당일 등락률 하한 (%)", -5.0, 2.0, -2.0, step=0.5)
        max_scan_count = st.number_input("최대 상세 조회 종목 수", 50, 300, 150, step=25,
                                         help="응답 속도를 위해 거래대금 상위 종목 중 이 수만 최근 시세를 조회합니다.")

    if st.button("주도 후보군 스크리닝 실행", type="primary"):
        with st.spinner("KRX 종목 스캔 및 20일 거래량·수익률 검증 중..."):
            try:
                df_krx = load_krx_listing()
                if df_krx.empty:
                    st.error("상장 종목 데이터를 수집하지 못했습니다.")
                else:
                    data_source = df_krx['DataSource'].iloc[0] if 'DataSource' in df_krx.columns else '원천 미확인'
                    df_krx = df_krx.drop_duplicates(subset=['Code']).reset_index(drop=True)
                    df_cap_filtered = df_krx[df_krx['Marcap'] >= (min_market_cap * 100000000)].copy()
                    df_cap_filtered = df_cap_filtered.sort_values(by="Amount", ascending=False)
                    percent_count = max(10, int(len(df_cap_filtered) * (top_amount_pct / 100)))
                    top_count = min(percent_count, int(max_scan_count))
                    df_target_pool = df_cap_filtered.head(top_count)
                    
                    screened_stocks = []
                    fail_count = 0
                    error_samples = []
                    scan_date = datetime.date.today().isoformat()
                    candidate_codes = tuple(df_target_pool['Code'].astype(str).str.zfill(6).tolist())

                    # 첫 실행은 최대 8개 동시 조회, 동일 대상·날짜의 재실행은 캐시 사용
                    metrics_rows = load_screening_metrics_batch(candidate_codes, scan_date)
                    metrics_by_code = {item['code']: item for item in metrics_rows}
                    
                    progress_bar = st.progress(0)
                    total_len = len(df_target_pool)

                    for idx, (_, row) in enumerate(df_target_pool.iterrows()):
                        progress_bar.progress((idx + 1) / total_len)
                        code = str(row['Code']).zfill(6)
                        name = row['Name']
                        metrics = metrics_by_code.get(code)
                        if not metrics or not metrics['ok']:
                            fail_count += 1
                            if len(error_samples) < 3:
                                detail = metrics.get('error', '조회 결과 없음') if metrics else '조회 결과 없음'
                                error_samples.append(f"{name}({code}): {detail}")
                            continue

                        if (metrics['return_5d'] >= min_return_5d and
                            metrics['vol_ratio_20d'] >= vol_mult_20d and
                            metrics['vol_ratio_prev'] >= vol_mult_prev and
                            metrics['return_1d'] >= min_daily_return):
                            screened_stocks.append({
                                "코드": code, "종목명": name, "현재가": int(metrics['last_close']),
                                "당일등락(%)": round(metrics['return_1d'], 2),
                                "5일수익률(%)": round(metrics['return_5d'], 2),
                                "20일평균대비(배)": round(metrics['vol_ratio_20d'], 2),
                                "시가총액(억)": int(row['Marcap'] / 100000000),
                                "거래대금(억)": int(row['Amount'] / 100000000)
                            })
                    
                    progress_bar.empty()
                    res_df = pd.DataFrame(screened_stocks)
                    st.session_state["screened_df"] = res_df
                    if not res_df.empty:
                        msg = f"필터링 완료! 주도 후보군 발굴: **{len(res_df)}개** (데이터 원천: {data_source}, 초기 풀: {len(df_krx)}종목, 상세 조회: {len(df_target_pool)}종목)"
                        if fail_count > 0:
                            msg += f" [제외/데이터 부족: {fail_count}건]"
                        st.success(msg)
                    else:
                        st.warning(f"조건에 부합하는 종목이 없습니다. (데이터 원천: {data_source}) 필터 조건을 완화해 보세요.")
                    
                    if error_samples:
                        with st.expander("⚠️ 스크리너 데이터 조회 제외/오류 샘플"):
                            for err in error_samples:
                                st.write(f"- {err}")
            except Exception as e:
                st.error(f"스크리너 실행 오류: {e}")

    if "screened_df" in st.session_state and not st.session_state["screened_df"].empty:
        df_display = st.session_state["screened_df"]
        st.dataframe(df_display, width='stretch')

        if st.button("테마 및 주도주 AI 분류 (Gemini)"):
            if not client:
                st.warning("사이드바에 Gemini API Key를 입력해야 AI 분류 기능을 사용할 수 있습니다.")
            else:
                stock_names = df_display["종목명"].tolist()
                theme_prompt = f"""
                당신은 시니어 퀀트 리서치 센터장입니다. 
                아래 주도 종목들의 산업 연관성과 수급 성격을 바탕으로 시장 주도 테마와 1등 대장주/수혜주를 분류하세요:
                [목록]: {', '.join(stock_names)}
                - 테마명, 거시 트리거, 1등 대장주, 후발 수혜주 형식으로 사실관계에 입각하여 명확히 작성하세요.
                """
                with st.spinner("AI 분석 중..."):
                    try:
                        res_text = generate_content_with_retry(client, theme_prompt)
                        st.markdown(res_text)
                    except Exception as e:
                        st.error(f"오류: {e}")

# -------------------------------------------------------------
# STEP 2: 퀀트 실행 분석 (Quad-Matrix)
# -------------------------------------------------------------
with tab2:
    st.subheader("🎯 STEP 2 · 퀀트 실행 분석 (Quad-Matrix)")
    st.caption("종목 퀄리티와 진입 타이밍을 엄격히 분리하고, R-Multiple 익절과 2단계 손절 규칙으로 기계적 매매를 지원합니다.")
    
    target_stock = st.text_input("분석할 종목명을 입력하세요 (예: JW홀딩스, jw홀딩스, SK하이닉스, 001060)")

    if st.button("종목 정밀 실행 분석 실행", type="primary"):
        if not target_stock.strip():
            st.warning("종목명을 입력해주세요.")
        else:
            with st.spinner(f"'{target_stock}' 종목 확인 및 Quad-Matrix 연산 중..."):
                try:
                    df_krx = load_krx_listing()
                    target_code, verified_name = find_stock_code(target_stock, df_krx)

                    if not target_code:
                        st.error(f"'{target_stock}' 종목을 찾을 수 없습니다. 정확한 종목명이나 6자리 코드를 확인하세요.")
                    else:
                        start_hist = (datetime.datetime.today() - datetime.timedelta(days=400)).strftime("%Y-%m-%d")
                        df_price = fdr.DataReader(target_code, start_hist)
                        
                        # 21거래일 전체에 대한 결측/양수 검증 (P2 해결)
                        required_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
                        if df_price is None or not all(col in df_price.columns for col in required_cols):
                            st.error(f"'{verified_name}'({target_code})의 OHLCV 필수 데이터가 누락되었습니다.")
                        elif len(df_price) < 25:
                            st.error(f"'{verified_name}'({target_code})의 과거 주가 데이터가 부족합니다 (최소 25거래일 필요).")
                        elif df_price.tail(21)[required_cols].isna().any().any():
                            st.error(f"'{verified_name}'의 최근 21거래일 시세에 결측치(NaN)가 포함되어 있습니다.")
                        elif (df_price.tail(21)[['Close', 'Open', 'High', 'Low']] <= 0).any().any():
                            st.error(f"'{verified_name}'의 최근 21거래일 시세에 비정상 가격(0 이하)이 존재합니다.")
                        else:
                            extra_data = get_comprehensive_stock_data(target_code)
                            q = run_quad_execution_engine(df_price, extra_data)

                            # UI 1. 상태 배너
                            if q["state_color"] == "warning":
                                st.warning(f"### 🟡 {q['market_state']}\n**행동 지침**: {q['trigger_action']}")
                            elif q["state_color"] == "success":
                                st.success(f"### 🟢 {q['market_state']}\n**행동 지침**: {q['trigger_action']}")
                            elif q["state_color"] == "error":
                                st.error(f"### 🔴 {q['market_state']}\n**행동 지침**: {q['trigger_action']}")
                            else:
                                st.info(f"### ⚪ {q['market_state']}\n**행동 지침**: {q['trigger_action']}")

                            # UI 2. QUAD-MATRIX
                            st.markdown("### 📊 4대 독립 판단 매트릭스 (Quad-Matrix)")
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("① QUALITY (퀄리티)", f"{q['quality_score']} / 100점", help="펀더멘털, 수급, 정배열 추세 종합 매력도")
                            c2.metric("② TIMING (진입자리)", f"{q['timing_score']} / 100점", help="20일선 근접도 및 눌림목 적합도")
                            c3.metric("③ RISK (과열/하방위험)", f"{q['risk_score']} / 100점", q['risk_label'], help="과열 및 저항 충돌 리스크")
                            c4.metric("④ TRIGGER (실행신호)", q['trigger_state'], f"스코어: {q['trig_score']}/100점")

                            ma60_str = f"{q['ma60']:,}원" if q['ma60'] else "미확인"
                            st.markdown(f"**핵심 지표 요약**: 현재가 **{q['curr_price']:,}원** | 20일선 이격도 **{q['disp_display_str']}** | RSI(14) **{q['rsi_14']} ({q['rsi_desc']})** | 20일 평균 대비 거래량 **{q['vol_ratio_20d']}배** | 60일선 **{ma60_str}**")

                            # UI 3. BUY ZONE & BREAKOUT PLAN
                            st.markdown("### 🎯 기계적 매수 계획 (Zone & Breakout)")
                            b_col1, b_col2 = st.columns(2)
                            with b_col1:
                                st.markdown("#### 🟢 눌림목 매수 (Pullback Plan)")
                                st.markdown(f"""
                                * **관찰 Buy Zone**: `{q['pullback_range']}` (기준가: {q['target_entry']:,}원)
                                * **원칙**: 가격 도달만으로 절대 매수 금지 (반등 확인 필수)
                                * **필수 반등 Trigger (2개 이상 충족 시 분할 진입 가동)**:
                                  - [ ] 20일선(±1.5%) 지지 확인 후 종가가 20일선 위 안착
                                  - [ ] 당일 양봉 마감 (종가 > 시가)
                                  - [ ] 장중 저점 찍고 전일 종가 이상 회복
                                  - [ ] 종가 기준 5일 이동평균선 재탈환
                                  - [ ] 거래량 20일 평균 대비 1.2배 이상 또는 외인 5일 순매수 유입
                                """)
                            with b_col2:
                                st.markdown("#### 🔵 전고점 돌파 매수 (Breakout Plan)")
                                st.markdown(f"""
                                * **돌파 타겟 기준선**: `{q['breakout_price']:,}원` (직전 20일 고점: {q['high_20']:,}원 상향 돌파)
                                * **돌파 확인 조건**:
                                  - [ ] **1차 확인**: 20일 평균 거래량 1.5배 이상 수반하며 종가 기준 돌파
                                  - [ ] **2차 확인**: 익일 재테스트 시 돌파선({q['high_20']:,}원) 종가 유지
                                """)

                            # UI 4. 손절선 구조화
                            st.markdown("### 🛑 리스크 관리: 2단계 손절선 구조")
                            st.caption(f"※ 초기 손절 손실률 기준가: **{q['target_entry']:,}원** (호가 단위 내림 적용)")
                            s1, s2, s3 = st.columns(3)
                            s1.metric("단기 초기 손절", q['stop_short_init'], "20일선 -1.5% 이탈")
                            s2.metric("스윙 초기 손절 (권장)", q['stop_swing_init'], "20일선 -3.0% 이탈")
                            s3.metric("중기 초기 손절", q['stop_mid_init'], "60일선 -3.0% 이탈")

                            # UI 5. R-Multiple 목표
                            st.markdown("### 📈 R-Multiple 익절 목표 vs 실제 차트 저항선")
                            st.caption(f"단위 리스크(1R) = **{q['risk_unit_R']:,}원** (예상 진입가 - 스윙 손절가 실측폭)")
                            
                            r_col1, r_col2 = st.columns(2)
                            with r_col1:
                                st.markdown("#### 🎯 R-Multiple 기반 기계적 익절")
                                st.markdown(f"""
                                * **목표 1R**: `{q['target_1R']}` (리스크 단위 1배 도달)
                                * **목표 2R (1차 권장 익절)**: `{q['target_2R']}` (신규 진입 성공 시 30% 차익 실현)
                                * **목표 3R (추세 확장)**: `{q['target_3R']}` (잔량 추세 트레일링 타겟)
                                """)
                            with r_col2:
                                st.markdown("#### 🧱 실제 차트 저항선 레이어")
                                st.markdown(f"""
                                * **직전 20일 전고점 저항**: `{q['high_20']:,}원` (매물 소화 점검선)
                                * **돌파 확정 기준선**: `{q['breakout_price']:,}원` (+0.5% 상향 안착)
                                * **52주 최고가 매물대**: `{q['high_52w']:,}원`
                                """)

                            # Gemini 브리핑 리포트 (키 존재 시에만 동작)
                            if client:
                                ai_prompt = f"""
                                당신은 리서치 센터의 수석 퀀트 애널리스트입니다.
                                제공된 정량 데이터를 기준으로 {verified_name}({target_code})의 매매 브리핑을 객관적으로 작성하세요.

                                [확정 정량 데이터]
                                - 종목명: {verified_name} ({target_code})
                                - 현재가: {q['curr_price']:,}원 (20일선 이격도: {q['disp_display_str']}, RSI: {q['rsi_14']} [{q['rsi_desc']}])
                                - 20일 평균 대비 거래량: {q['vol_ratio_20d']}배
                                - 수급 데이터 신뢰도: {extra_data['data_confidence']}
                                - ① Quality: {q['quality_score']}/100
                                - ② Timing: {q['timing_score']}/100
                                - ③ Risk: {q['risk_score']}/100 ({q['risk_label']})
                                - ④ Trigger: {q['trigger_state']} (스코어: {q['trig_score']}/100점)
                                - 현재 상태: {q['market_state']}
                                - 눌림목 Buy Zone: {q['pullback_range']} (예상 진입 기준가: {q['target_entry']:,}원)
                                - 초기 스윙 손절선: {q['stop_swing_init']}
                                - 2R 익절 목표가: {q['target_2R']}
                                - 직전 20일 전고점 저항선: {q['high_20']:,}원 (돌파 기준: {q['breakout_price']:,}원)

                                [작성 규격]
                                ### 1. Quad-Matrix 정량 평가 및 위치 진단
                                ### 2. 거래량 및 가격 구조 객관적 해석
                                ### 3. 조건부 매매 실행 시나리오
                                ### 4. R-Multiple 익절 및 2단계 손절선 관리
                                """

                                with st.spinner("Gemini AI가 정밀 매매 실행 리포트를 작성하고 있습니다..."):
                                    report_text = generate_content_with_retry(client, ai_prompt)
                                    st.markdown(report_text)
                            else:
                                st.info("💡 사이드바에 Gemini API Key를 등록하면 AI 심층 분석 브리핑 리포트가 함께 생성됩니다.")

                except Exception as e:
                    st.error(f"분석 중 오류 발생: {e}")

# -------------------------------------------------------------
# STEP 3: 사용 매뉴얼
# -------------------------------------------------------------
with tab3:
    st.subheader("📖 QUANT-EXECUTION 시스템 매뉴얼")
    st.markdown("""
    ### 💡 4대 독립 판단 체계 (Quad-Matrix)
    * **① QUALITY (0~100점)**: 펀더멘털, 수급, 추세 등 종목 자체의 체력 평가
    * **② TIMING (0~100점)**: 20일선 근접도 및 눌림목 적합도 (추격매수 패널티 부여)
    * **③ RISK (0~100점)**: 과열도(RSI, 이격도) 및 저항 충돌 위험도
    * **④ TRIGGER (ACTIVE / WATCH / WAIT / OFF)**: 기계적 진입 조건 충족 여부 (Buy Zone 내 2개 조건 충족 시 ACTIVE 분할 매수 가동)

    ### 🛑 2단계 리스크 관리
    * **초기 구조적 손절선**: 진입 시점 기준 -1.5% ~ -3.0% 사전 설정 (호가 단위 보수적 내림 적용)
    * **동적 트레일링 손절선**: 주가 상승 시 20일선 상향에 맞춰 손절선 동반 추종
    """)
