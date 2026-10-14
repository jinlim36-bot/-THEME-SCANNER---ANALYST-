import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import io
import math
import logging
import FinanceDataReader as fdr

# =============================================================
# [시스템 로깅 & 환경 설정]
# =============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("QuantExecutionProMax")

# KRX 주식 호가 단위: 각 구간 상한은 미포함 (ETF/ETN에는 별도 규칙 적용)
KOREA_TICK_TABLE = [
    (2000, 1),
    (5000, 5),
    (20000, 10),
    (50000, 50),
    (200000, 100),
    (500000, 500),
    (float('inf'), 1000)
]

# 대표 섹터 ETF 맵핑 (3단 계층 상대강도 산출용)
SECTOR_ETF_MAP = {
    "반도체": "091160",      # KODEX 반도체
    "2차전지": "305720",     # KODEX 2차전지산업
    "바이오": "244580",      # KODEX 바이오
    "자동차": "091180",      # KODEX 자동차
    "IT/플랫폼": "229200",   # KODEX 코스닥150: IT 전용 ETF가 아닌 대용 지표
    "대표지수": "069500"     # KODEX 200
}

# KRX 표준 산업 분류 -> 5대 주도 섹터 ETF 매핑 규칙
KRX_INDUSTRY_KEYWORD_MAP = {
    "반도체": "반도체",
    "이차전지": "2차전지", "2차전지": "2차전지", "축전지": "2차전지",
    "의약": "바이오", "바이오": "바이오", "약품": "바이오", "생물": "바이오",
    "자동차": "자동차", "트레일러": "자동차",
    "소프트웨어": "IT/플랫폼", "인터넷": "IT/플랫폼", "포털": "IT/플랫폼", "게임": "IT/플랫폼"
}

st.set_page_config(
    page_title="QUANT-EXECUTION PRO MAX: 완전 일치 실행 엔진",
    page_icon="⚖️",
    layout="wide"
)

@st.cache_data(ttl=3600)
def load_price_history(symbol, start_date):
    try:
        data = fdr.DataReader(symbol, start_date)
        if data is None or data.empty:
            return pd.DataFrame()
        data = data.copy()
        data.attrs['retrieved_at'] = market_now().isoformat()
        data.index = pd.to_datetime(data.index).normalize()
        return data.loc[~data.index.duplicated(keep="last")].sort_index()
    except Exception as error:
        logger.warning("Price data unavailable [%s]: %s", symbol, type(error).__name__)
        st.warning(f"{symbol} 시세 조회 실패 ({type(error).__name__}). 데이터 제공처와 네트워크 설정을 확인하세요.")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_stock_listing(market):
    try:
        data = fdr.StockListing(market)
        return data if data is not None else pd.DataFrame()
    except Exception as error:
        logger.warning("Stock listing unavailable [%s]: %s", market, type(error).__name__)
        st.warning(f"{market} 종목 목록 조회 실패 ({type(error).__name__}).")
        return pd.DataFrame()


# -------------------------------------------------------------
# 1. 호가 단위 및 가격 헬퍼
# -------------------------------------------------------------
def get_tick_size(price):
    try:
        p = abs(float(price))
        for threshold, tick in KOREA_TICK_TABLE:
            if p < threshold:
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
def prepare_engine_history(df):
    """Compute causal indicators once; every row depends only on earlier rows."""
    if '_engine_atr14' in df.columns:
        return df
    result = df.copy()
    close = result['Close']
    result['_engine_ma5'] = close.rolling(5).mean()
    result['_engine_ma20'] = close.rolling(20).mean()
    result['_engine_ma20_prev5'] = result['_engine_ma20'].shift(5)
    result['_engine_ma60'] = close.rolling(60).mean()
    previous = close.shift(1)
    tr = pd.concat([result['High'] - result['Low'],
                    (result['High'] - previous).abs(),
                    (result['Low'] - previous).abs()], axis=1).max(axis=1)
    result['_engine_atr14'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    return result


def calculate_wilder_atr(df, period=14):
    if len(df) < period + 1: return 0.0
    if period == 14 and "_engine_atr14" in df.columns:
        atr = df["_engine_atr14"].iloc[-1]
        return float(atr) if pd.notna(atr) else 0.0
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
def valid_price_history(df, min_rows=25):
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    if df is None or len(df) < min_rows or not df.columns.is_unique or not all(c in df for c in required):
        return False
    try:
        values = df[required].tail(min_rows).to_numpy(dtype=float)
    except (TypeError, ValueError):
        return False
    o, h, low, close, volume = values.T
    return bool(np.isfinite(values).all() and (values[:, :4] > 0).all()
                and (volume >= 0).all() and (h >= np.maximum.reduce([o, close, low])).all()
                and (low <= np.minimum.reduce([o, close, h])).all())


def benchmark_is_current(stock, benchmark, min_rows=21):
    if stock is None or len(stock) < min_rows or benchmark is None or benchmark.empty or 'Close' not in benchmark:
        return False
    try:
        # Require the exact recent trading dates; do not stretch across missing rows.
        values = benchmark['Close'].reindex(stock.index[-min_rows:]).to_numpy(dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(values).all() and (values > 0).all())


def regime_at(df_regime, date):
    # A missing trading-day observation is unknown, not a neutral market.
    if df_regime is None or df_regime.empty or date not in df_regime.index:
        return 'UNKNOWN'
    row = df_regime.loc[date]
    code = row.get('Regime', 'UNKNOWN')
    close = row.get('KS_Close', np.nan)
    return code if code in {'FAVORABLE', 'NEUTRAL', 'RISK-OFF'} and pd.notna(close) and close > 0 else 'UNKNOWN'


def build_dual_regime(df_ks, df_kq):
    if df_ks is None or df_kq is None or 'Close' not in df_ks or 'Close' not in df_kq:
        return pd.DataFrame()
    common = df_ks.index.intersection(df_kq.index).sort_values()
    if len(common) < 30:
        return pd.DataFrame()
    ks = pd.to_numeric(df_ks.loc[common, 'Close'], errors='coerce')
    kq = pd.to_numeric(df_kq.loc[common, 'Close'], errors='coerce')
    m_ks, m_kq = ks.rolling(20).mean(), kq.rolling(20).mean()
    slope_ks, slope_kq = m_ks.pct_change(5, fill_method=None), m_kq.pct_change(5, fill_method=None)
    valid = np.isfinite(pd.concat([ks, kq, m_ks, m_kq, slope_ks, slope_kq], axis=1)).all(axis=1)
    valid &= (ks > 0) & (kq > 0) & (m_ks > 0) & (m_kq > 0)
    bull = (ks >= m_ks) & (slope_ks > 0) & (kq >= m_kq) & (slope_kq > 0)
    bear = ((ks < m_ks) & (slope_ks <= 0)) | ((kq < m_kq) & (slope_kq <= 0))
    codes = np.select([~valid, bear, bull], ['UNKNOWN', 'RISK-OFF', 'FAVORABLE'], default='NEUTRAL')
    result = pd.DataFrame({'Regime': codes, 'KS_Close': ks, 'KQ_Close': kq}, index=common)
    snapshots = [df.attrs.get('retrieved_at') for df in (df_ks, df_kq)]
    if all(snapshots):
        result.attrs['retrieved_at'] = min(snapshots)
    return result


@st.cache_data(ttl=3600)
def load_historical_dual_regime(start_date="2022-01-01"):
    try:
        return build_dual_regime(load_price_history('KS11', start_date), load_price_history('KQ11', start_date))
    except Exception as error:
        logger.error("Dual regime unavailable: %s", type(error).__name__)
        return pd.DataFrame()

def market_now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))


def market_today():
    return market_now().date()


def completed_live_history(data, now=None):
    """Only live views use the 15:40 KST completion buffer; historical engines do not."""
    if data is None or data.empty:
        return pd.DataFrame()
    now = now or market_now()
    snapshot = pd.Timestamp(data.attrs.get('retrieved_at', now))
    if snapshot.tzinfo is None:
        snapshot = snapshot.tz_localize(now.tzinfo)
    snapshot = snapshot.tz_convert(now.tzinfo)
    cutoff = min(now, snapshot.to_pydatetime())
    last_day = cutoff.date()
    if cutoff.time().replace(tzinfo=None) < datetime.time(15, 40):
        last_day -= datetime.timedelta(days=1)
    return data.loc[data.index.normalize() <= pd.Timestamp(last_day)].copy()


def live_snapshot_issue(stock, market, regime_date):
    if stock is None or stock.empty or market is None or market.empty:
        return '확정 종목·시장 시세가 부족합니다.'
    stock_date, market_date = stock.index[-1].date(), market.index[-1].date()
    age = (market_today() - market_date).days
    if age < 0 or age > 7:
        return '시장 시세가 미래 날짜이거나 7일을 초과했습니다.'
    if stock_date != market_date:
        return '종목의 마지막 거래일이 최신 시장 거래일과 다릅니다. 거래정지·시세 누락을 확인하세요.'
    if regime_date and str(market_date) != regime_date:
        return '시장 국면과 비교 지수의 확정 기준일이 다릅니다. 시세·시장 데이터를 새로 조회하세요.'
    return None


def evaluate_live_execution(stock, regime, market, sector=None, has_frgn=False):
    stock, market = completed_live_history(stock), completed_live_history(market)
    sector = completed_live_history(sector)
    issue = live_snapshot_issue(stock, market, regime.get('data_date'))
    if issue:
        return blocked_decision(issue), stock
    return evaluate_full_execution_pipeline(stock, regime['code'], kospi_slice=market,
                                            sector_slice=sector, has_frgn=has_frgn), stock


def get_current_dual_regime():
    today = market_today()
    df = load_historical_dual_regime((today - datetime.timedelta(days=90)).isoformat())
    unknown = {'status': 'UNKNOWN (시장 데이터 미확인)', 'code': 'UNKNOWN', 'multiplier': 0.0,
               'msg': '시장 데이터가 없거나 기준일이 7일을 초과해 신규 매수를 보류합니다.', 'data_date': None}
    if df.empty:
        return unknown
    if df.index[-1].date() > today:
        return unknown
    df = completed_live_history(df)
    if df.empty:
        return unknown
    date = df.index[-1]
    unknown['data_date'] = date.strftime('%Y-%m-%d')
    age = (today - date.date()).days
    code = regime_at(df, date)
    if age < 0 or age > 7 or code == 'UNKNOWN':
        return unknown
    states = {
        'FAVORABLE': ('FAVORABLE (양대 지수 우호)', 1.0, '양대 지수 상승 추세. 종목 조건과 손실 한도를 확인합니다.'),
        'RISK-OFF': ('RISK-OFF (시장 위험 경보)', 0.0, '신규 매수를 금지합니다. 기존 보유분은 별도 청산 규칙을 따릅니다.'),
        'NEUTRAL': ('NEUTRAL (경계/혼조 국면)', 0.5, '신규 매수 손실 예산을 50%로 축소합니다.'),
    }
    status, multiplier, msg = states[code]
    return {'status': status, 'code': code, 'multiplier': multiplier, 'msg': msg,
            'data_date': date.strftime('%Y-%m-%d')}

# -------------------------------------------------------------
# 4. 데이터 로더 & 종목 매핑 및 자동 섹터 분류 엔진
# -------------------------------------------------------------
@st.cache_data(ttl=86400)
def load_all_krx_master():
    try:
        url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        if res.status_code == 200:
            dfs = pd.read_html(io.BytesIO(res.content), encoding="cp949")
            if dfs and not dfs[0].empty:
                df = dfs[0][['회사명', '종목코드', '업종']].copy() if '업종' in dfs[0].columns else dfs[0][['회사명', '종목코드']].copy()
                df.columns = ['Name', 'Code', 'Sector_Raw'] if '업종' in dfs[0].columns else ['Name', 'Code']
                df['Code'] = df['Code'].astype(str).str.zfill(6)
                return df.drop_duplicates(subset=['Code']).reset_index(drop=True)
    except Exception as e:
        logger.error(f"KIND 로드 실패: {e}")
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_krx_listing():
    try:
        df = load_stock_listing('KRX')
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

def build_sector_lookup(df_krx=None):
    master = load_all_krx_master()
    raw_sectors = {}
    for data, column in ((df_krx, 'Sector'), (master, 'Sector_Raw')):
        if data is not None and not data.empty and {'Code', column}.issubset(data.columns):
            for code, sector in zip(data['Code'], data[column]):
                if pd.notna(sector) and str(sector).strip():
                    raw_sectors[str(code).zfill(6)] = str(sector)
    return {code: next((mapped for keyword, mapped in KRX_INDUSTRY_KEYWORD_MAP.items()
                        if keyword in sector), '대표지수')
            for code, sector in raw_sectors.items()}


def get_sector_for_stock(code, df_krx=None, sector_lookup=None):
    """Use one industry snapshot per scan; unknown industries use the index ETF."""
    if sector_lookup is None:
        sector_lookup = build_sector_lookup(df_krx)
    return sector_lookup.get(str(code).zfill(6), '대표지수')


def get_supply_data(code, as_of_date=None):
    headers = {"User-Agent": "Mozilla/5.0"}
    res_data = {"frgn_5": None, "status": "FAIL"}
    try:
        url = f"https://finance.naver.com/item/frgn.naver?code={code}"
        res = requests.get(url, headers=headers, timeout=4)
        if res.status_code == 200:
            dfs = pd.read_html(io.BytesIO(res.content), encoding="cp949")
            df_frgn = next((t for t in dfs if any("외국인" in str(c) and "순매" in str(c) for c in t.columns)), None)
            if df_frgn is not None:
                df_frgn.columns = ['_'.join(c).strip() if isinstance(c, tuple) else str(c) for c in df_frgn.columns]
                f_cols = [c for c in df_frgn.columns if "외국인" in c and "순매" in c]
                if f_cols:
                    flows = pd.to_numeric(df_frgn[f_cols[0]].astype(str).str.replace(',', ''), errors='coerce')
                    if as_of_date is not None:
                        date_columns = [c for c in df_frgn.columns if '날짜' in c]
                        if not date_columns:
                            return res_data
                        dates = pd.to_datetime(df_frgn[date_columns[0]], errors='coerce')
                        dated = pd.DataFrame({'Date': dates, 'Flow': flows}).dropna().drop_duplicates('Date')
                        dated = dated[dated['Date'] <= pd.Timestamp(as_of_date).normalize()].sort_values('Date', ascending=False)
                        if dated.empty or dated['Date'].iloc[0] != pd.Timestamp(as_of_date).normalize():
                            return res_data
                        flows = dated['Flow']
                    fv = flows.dropna().tolist()
                    if len(fv) >= 5:
                        res_data["frgn_5"] = int(sum(fv[:5]))
                        res_data["status"] = "OK"
    except Exception as error:
        logger.warning('Foreign supply unavailable [%s]: %s', code, type(error).__name__)
    return res_data

@st.cache_data(ttl=86400)
def load_historical_foreign_series(code, pages=18):
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
    return total_df['Net_Frgn'].rolling(5).sum()

def load_sector_etf_series(sector_name, start_date="2022-01-01"):
    etf_code = SECTOR_ETF_MAP.get(sector_name, "069500")
    try:
        df = load_price_history(etf_code, start_date)
        if df is not None and not df.empty:
            return df[['Close']]
    except Exception as error:
        logger.warning('Sector ETF data unavailable [%s]: %s', etf_code, type(error).__name__)
    return pd.DataFrame()

# -------------------------------------------------------------
# 5. [3단 계층 Sector RS 및 100점 Entry Score 엔진]
# -------------------------------------------------------------
def calculate_3tier_rs(df_slice, kospi_slice, sector_slice=None):
    if not benchmark_is_current(df_slice, kospi_slice):
        return 0, '시장 비교 데이터 미확인 (가점 없음)'
    sector_available = benchmark_is_current(df_slice, sector_slice)
    dates = df_slice.index[-21:]
    stk = pd.to_numeric(df_slice.loc[dates, 'Close'], errors='coerce')
    market = pd.to_numeric(kospi_slice.loc[dates, 'Close'], errors='coerce')
    if len(dates) < 21 or not np.isfinite(stk).all() or (stk <= 0).any():
        return 0, '비교 데이터 부족 (가점 없음)'
    stock_return = (stk.iloc[-1] / stk.iloc[0] - 1) * 100
    market_return = (market.iloc[-1] / market.iloc[0] - 1) * 100
    stock_market = stock_return - market_return
    if not sector_available:
        points = 10 if stock_market >= 5 else 6 if stock_market >= 0 else 0
        return points, f'섹터 미확인 · 종목-시장 {stock_market:+.1f}%p'
    sector = pd.to_numeric(sector_slice.loc[dates, 'Close'], errors='coerce')
    if not np.isfinite(sector).all() or (sector <= 0).any():
        return 0, '섹터 비교 데이터 오류 (가점 없음)'
    sector_return = (sector.iloc[-1] / sector.iloc[0] - 1) * 100
    sector_market = sector_return - market_return
    stock_sector = stock_return - sector_return
    description = (f'종목-시장 {stock_market:+.1f}%p / 섹터-시장 {sector_market:+.1f}%p'
                   f' / 종목-섹터 {stock_sector:+.1f}%p')
    if stock_market >= 5 and sector_market >= 2 and stock_sector >= 0:
        return 15, '주도섹터 내 우위 · ' + description
    if stock_market >= 0 and sector_market >= 0 and stock_sector >= 0:
        return 10, '동반 강세 · ' + description
    if stock_market >= 0 and sector_market < 0:
        return 6, '소외섹터 내 반등 · ' + description
    if stock_market >= 0:
        return 5, '섹터 대비 약세 · ' + description
    return 0, '시장 대비 약세 · ' + description

def detect_support_rebound(df_slice):
    """MA20 support, tested in the last three bars; no future bars are read."""
    result = {'state': 'UNVERIFIED', 'label': '패턴 미확인', 'reason': '전일 MA60 기준 확보에 최소 61거래일이 필요합니다.'}
    if not valid_price_history(df_slice, min_rows=61):
        return result
    data = prepare_engine_history(df_slice)
    previous, bar = data.iloc[-2], data.iloc[-1]
    support, atr = float(previous['_engine_ma20']), float(previous['_engine_atr14'])
    if not np.isfinite([support, atr, previous['_engine_ma60'], previous['_engine_ma20_prev5']]).all() or atr <= 0:
        return result
    recent = data.iloc[-3:]
    # Each test must use the support that was known before THAT candle, rather
    # than moving today's MA20 backwards onto earlier candles.
    references = data['_engine_ma20'].shift(1).iloc[-3:]
    prior_atrs = data['_engine_atr14'].shift(1).iloc[-3:]
    lower_limits = (references - .75 * prior_atrs).map(floor_to_tick)
    touches = ((recent['Low'] <= references + .35 * prior_atrs)
               & (recent['High'] >= references - .35 * prior_atrs)
               & (recent['Low'] > lower_limits))
    reference_date = data.index[-2]
    if touches.any():
        touch_date = recent.index[touches][-1]
        touch_index = data.index.get_loc(touch_date)
        support = float(data['_engine_ma20'].iloc[touch_index-1])
        atr = float(data['_engine_atr14'].iloc[touch_index-1])
        reference_date = data.index[touch_index-1]
    band = .35 * atr
    invalidation = floor_to_tick(support - .75 * atr)
    result.update(support=round(support, 2), band=round(band, 2), atr=atr,
                  invalidation_price=invalidation, signal_date=data.index[-1].strftime('%Y-%m-%d'),
                  reference_date=reference_date.strftime('%Y-%m-%d'))
    def state(code, label, reason):
        return dict(result, state=code, label=label, reason=reason)
    close = float(bar['Close'])
    if close <= invalidation:
        return state('BROKEN', '지지 이탈', '종가가 지지 무효화 가격 아래입니다.')
    if (float(previous['_engine_ma20']) < float(previous['_engine_ma60'])
            or float(previous['_engine_ma20']) <= float(previous['_engine_ma20_prev5'])):
        return state('NO_SETUP', '추세 조건 미달', '전일 MA20이 MA60 이상이고 최근 5일 상승하는 추세가 필요합니다.')
    if close > support + 1.5 * atr:
        return state('EXTENDED', '지지선과 거리 과다', '종가가 지지선보다 1.5 ATR 이상 높아 추격 진입을 보류합니다.')
    if not touches.any():
        if abs(close - support) <= 1.5 * atr:
            return state('APPROACH', '지지 접근', '지지 구간의 최근 시험이 없어 관찰합니다.')
        return state('NO_SETUP', '지지 시험 없음', '최근 3개 봉에서 지지 구간을 시험하지 않았습니다.')
    touch_date = recent.index[touches][-1]
    result['touch_date'] = touch_date.strftime('%Y-%m-%d')
    if (recent.loc[touch_date:, 'Low'] <= invalidation).any():
        return state('BROKEN', '지지 이탈', '지지 시험 이후 무효화 가격을 침범했습니다.')
    if close < support:
        return state('APPROACH', '지지 회복 대기', '종가가 지지선 위로 회복해야 합니다.')
    candle_range = float(bar['High'] - bar['Low'])
    close_position = (close - float(bar['Low'])) / candle_range if candle_range > 0 else 0
    volume_ratio = float(bar['Volume']) / get_clean_avg_volume(data['Volume'], 20)
    if not (close > float(bar['Open']) and close > float(previous['Close'])
            and close_position >= .65 and volume_ratio >= .9):
        return state('RECOVERED', '지지 회복 · 반등 대기', '양봉·전일 종가 상승·봉 상단 35% 종가·평균 거래량 90% 이상을 기다립니다.')
    trigger = ceil_to_tick(float(bar['High']) + get_tick_size(float(bar['High'])))
    result.update(trigger_price=trigger, max_entry_price=floor_to_tick(trigger + .5 * atr))
    return state('CONFIRMED', '반등 확인 · 돌파 대기', '반등 봉 고가 위의 트리거를 다음 거래일에 돌파할 때만 조건부 진입합니다.')


def calculate_100p_score(df_slice, kospi_slice=None, sector_slice=None, has_frgn=False):
    if len(df_slice) < 25:
        return {"score": 0, "grade": "AVOID", "details": {}, "track": "WAIT"}

    curr_p = float(df_slice['Close'].iloc[-1])
    open_p = float(df_slice['Open'].iloc[-1])
    high_p = float(df_slice['High'].iloc[-1])
    low_p = float(df_slice['Low'].iloc[-1])

    ma5 = float(df_slice["_engine_ma5"].iloc[-1]) if "_engine_ma5" in df_slice.columns else float(df_slice['Close'].iloc[-5:].mean())
    ma20 = float(df_slice["_engine_ma20"].iloc[-1]) if "_engine_ma20" in df_slice.columns else float(df_slice['Close'].iloc[-20:].mean())
    ma20_5d = float(df_slice["_engine_ma20_prev5"].iloc[-1]) if "_engine_ma20_prev5" in df_slice.columns else float(df_slice['Close'].iloc[-25:-5].mean())
    ma60 = (float(df_slice["_engine_ma60"].iloc[-1]) if "_engine_ma60" in df_slice.columns else float(df_slice['Close'].iloc[-60:].mean())) if len(df_slice) >= 60 else ma20

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
    details["MA20이격"] = f"{disp_pts}/20"

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

    # 5. 3단 계층 상대강도 RS (15점 만점 실질 연결)
    rs_pts, rs_desc = calculate_3tier_rs(df_slice, kospi_slice, sector_slice)
    score += rs_pts
    details["상대강도(RS)"] = f"{rs_pts}/15 ({rs_desc})"

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

    # A score alone does not select an entry. The pattern gate assigns REBOUND.
    track = "WAIT"

    return {
        "score": score, "grade": grade, "details": details,
        "track": track, "disparity_pct": disparity_pct,
        "high_20": int(high_20), "ma20": int(ma20), "ma20_slope_pct": ma20_slope_pct
    }

# -------------------------------------------------------------
# 6. 가격선 설계 & 과도 손절폭 필터 (정합적 산출식 적용)
# -------------------------------------------------------------
def calculate_execution_levels(df_slice, entry_price, invalidation_price=None):
    atr14 = calculate_wilder_atr(df_slice, 14)
    curr_p = float(entry_price)
    ma20 = float(df_slice["_engine_ma20"].iloc[-1]) if "_engine_ma20" in df_slice.columns else float(df_slice['Close'].iloc[-20:].mean())
    recent_swing_low = float(df_slice['Low'].iloc[-10:-1].min())
    high_20 = float(df_slice['High'].iloc[-21:-1].max())

    # 첫 번째 무너지는 가장 가까운 지지선 후보 탐색
    stop_atr = curr_p - (1.5 * atr14)
    stop_struct = recent_swing_low - (0.2 * atr14)
    stop_trend = ma20 * 0.97
    candidate_stop = max(stop_atr, stop_struct, stop_trend)

    # 최소 1.2 ATR 안전거리 버퍼 확보
    min_noise_boundary = curr_p - (1.2 * atr14)
    final_stop_val = min(candidate_stop, min_noise_boundary)
    if invalidation_price is not None:
        final_stop_val = min(final_stop_val, invalidation_price)

    stop_price = floor_to_tick(final_stop_val)
    if stop_price >= curr_p:
        stop_price = floor_to_tick(curr_p * 0.96)

    risk_1r = curr_p - stop_price
    loss_pct = round(((stop_price - curr_p) / curr_p) * 100, 2)
    is_excessive_stop = (curr_p - stop_price) / curr_p >= 0.085

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
        "atr14": float(atr14)
    }

def blocked_decision(reason, pattern=None):
    scored = {'score': 0, 'grade': 'UNVERIFIED', 'details': {}, 'track': 'WAIT', 'ma20': 0, 'high_20': 0}
    levels = {key: 0 for key in ['entry_price', 'stop_price', 'loss_pct', 'risk_1r', 'target_1r',
                                 'target_2r', 'target_3r', 'high_20', 'rrr_to_resist', 'atr14']}
    levels.update(is_excessive_stop=True, resistance_warning=None)
    return {'action': 'WAIT', 'badge': '⚪ WAIT (데이터 미확인)', 'reason': reason,
            'scored': scored, 'levels': levels, 'multiplier': 0.0, 'block_code': 'DATA',
            'pattern': pattern or {'state': 'UNVERIFIED', 'label': '패턴 미확인', 'reason': reason}}


def evaluate_execution_features(df_slice, regime_code, kospi_slice=None, sector_slice=None, has_frgn=False):
    """Compute immutable per-date pattern and score features for all WFO thresholds."""
    if not valid_price_history(df_slice, min_rows=61):
        return blocked_decision('전일 MA60(60거래일) 기준선을 확보할 최소 61거래일의 유효 OHLCV 데이터가 필요합니다.')
    data = prepare_engine_history(df_slice)
    pattern = detect_support_rebound(data)
    if regime_code not in {'FAVORABLE', 'NEUTRAL', 'RISK-OFF'}:
        return blocked_decision('시장 상태를 확인할 수 없어 신규 매수를 보류합니다.', pattern)
    if not benchmark_is_current(data, kospi_slice):
        return blocked_decision('종목과 동일한 기준일의 시장 비교 데이터가 부족합니다.', pattern)
    scored = calculate_100p_score(data, kospi_slice, sector_slice=sector_slice, has_frgn=has_frgn)
    scored['track'] = 'REBOUND' if pattern['state'] == 'CONFIRMED' else 'WAIT'
    entry = pattern.get('trigger_price', int(data['Close'].iloc[-1]))
    levels = calculate_execution_levels(data, entry, pattern.get('invalidation_price'))
    ceiling = pattern.get('max_entry_price', entry)
    if ceiling > 0 and (ceiling - levels['stop_price']) / ceiling >= .085:
        levels['is_excessive_stop'] = True
    if levels['atr14'] <= 0:
        return blocked_decision('변동성 산출 데이터가 부족하여 신규 매수를 보류합니다.', pattern)
    return {'scored': scored, 'levels': levels, 'pattern': pattern}


def decide_execution(features, regime_code, min_score=70):
    """Pattern, market permission and execution eligibility are distinct states."""
    if 'action' in features:
        return dict(features)
    scored, levels, pattern = features['scored'], features['levels'], features['pattern']
    result = dict(features, action='WAIT', multiplier=0.0)
    if regime_code == 'RISK-OFF':
        return dict(result, action='AVOID', badge='🔴 진입 차단 (시장 위험)', block_code='MARKET',
                    reason='시장 RISK-OFF: 패턴은 관찰하되 신규 매수 주문을 내지 않습니다.')
    if pattern['state'] != 'CONFIRMED':
        return dict(result, badge='🟡 ' + pattern['label'], reason=pattern['reason'], block_code='PATTERN')
    if scored['score'] < min_score:
        return dict(result, badge=f"🟡 관찰 ({scored['score']}점)", reason=f'기준 점수({min_score}점) 미달', block_code='SCORE')
    if levels['is_excessive_stop']:
        return dict(result, badge='🟡 WAIT (손절폭 과도)', reason='허용 진입가 범위의 보수적 손절폭이 8.5% 이상입니다.', block_code='STOP')
    reduced = regime_code == 'NEUTRAL'
    return dict(result, action='BUY_REDUCED' if reduced else 'BUY', multiplier=.5 if reduced else 1.0,
                badge=f"{'🟡' if reduced else '🟢'} 조건부 진입 ({scored['score']}점{' · 비중 50%' if reduced else ''})",
                reason='다음 거래일 트리거 돌파 시 진입. 무효화 가격 침범·최대 진입가 초과·미돌파 시 주문을 취소합니다.', block_code='ELIGIBLE')


def evaluate_full_execution_pipeline(df_slice, regime_code, kospi_slice=None, sector_slice=None, has_frgn=False, min_score=70):
    features = evaluate_execution_features(df_slice, regime_code, kospi_slice, sector_slice, has_frgn)
    return decide_execution(features, regime_code, min_score)


def expected_stop_risk(entry_price, stop_price, buy_fee=0.0005, sell_fee=0.0020, market_slippage=0.0010):
    """Planned unit loss including costs; gaps beyond the stop are not bounded."""
    stop_execution = floor_to_tick(max(0, stop_price) * (1 - market_slippage))
    return max(0.0, entry_price * (1 + buy_fee) - stop_execution * (1 - sell_fee))


def historic_correlation(left, right, date, lookback=60, min_observations=30):
    # Today's high/low/close cannot influence today's allocation.
    a = left.loc[left.index < date, 'Close'].tail(lookback + 1).pct_change(fill_method=None)
    b = right.loc[right.index < date, 'Close'].tail(lookback + 1).pct_change(fill_method=None)
    aligned = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(aligned) < min_observations or (aligned.std() == 0).any():
        return None
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def limit_portfolio_entry(desired_shares, code, sector, entry_price, stop_price, equity,
                          positions, histories, date, max_total_risk_pct=3.0,
                          max_sector_risk_pct=2.0, max_correlated_risk_pct=2.0,
                          correlation_threshold=0.8, buy_fee=0.0005, sell_fee=0.0020,
                          market_slippage=0.0010):
    if equity <= 0 or desired_shares <= 0:
        return {'shares': 0, 'total_risk_pct': 0.0, 'limiting_factor': 'NO_RISK_BUDGET'}
    unit = expected_stop_risk(entry_price, stop_price, buy_fee, sell_fee, market_slippage)
    if unit <= 0 or not np.isfinite(unit):
        return {'shares': 0, 'total_risk_pct': 0.0, 'limiting_factor': 'INVALID_STOP_RISK'}
    total_used = sector_used = correlated_used = 0.0
    for existing_code, position in positions.items():
        opening_price = float(histories[existing_code].loc[date, 'Open'])
        entered_today = position['entry_date'] == date.strftime('%Y-%m-%d')
        if entered_today:
            # A position bought intraday did not exist at the earlier opening price.
            mark = position['entry_price'] * (1 + buy_fee)
            stop_basis = position['current_stop']
        else:
            mark = opening_price
            # An overnight stop gapped through fills at the open, with costs.
            stop_basis = min(opening_price, position['current_stop'])
        stop_net = floor_to_tick(stop_basis * (1 - market_slippage)) * (1 - sell_fee)
        used = position['shares'] * max(0, mark - stop_net)
        total_used += used
        if position.get('sector', '대표지수') == sector:
            sector_used += used
        corr = historic_correlation(histories[code], histories[existing_code], date)
        # Unavailable correlations are treated conservatively for the group budget.
        if corr is None or corr >= correlation_threshold:
            correlated_used += used
    caps = {
        'PORTFOLIO_RISK_CAP': max(0, equity * max_total_risk_pct / 100 - total_used),
        'SECTOR_RISK_CAP': max(0, equity * max_sector_risk_pct / 100 - sector_used),
        'CORRELATION_RISK_CAP': max(0, equity * max_correlated_risk_pct / 100 - correlated_used),
    }
    factor = min(caps, key=caps.get)
    allowed = max(0, int(caps[factor] // unit))
    shares = min(desired_shares, allowed)
    return {'shares': shares, 'total_risk_pct': (total_used + shares * unit) / equity * 100,
            'limiting_factor': factor if shares < desired_shares else 'PER_TRADE_LIMIT'}


def summarize_trades(trades, min_sample=30):
    count = len(trades)
    pnl = pd.to_numeric(trades['실현손익(원)'], errors='raise') if count else pd.Series(dtype=float)
    wins, losses = pnl[pnl > 0], -pnl[pnl < 0]
    if losses.sum() > 0:
        factor, state = float(wins.sum() / losses.sum()), 'finite'
    elif wins.sum() > 0:
        factor, state = float('inf'), 'no_losses'
    else:
        factor, state = float('nan'), 'no_trades' if not count else 'all_breakeven'
    def rate(column):
        return round(trades[column].astype(bool).sum() / count * 100, 1) if count and column in trades else 0.0
    return {'profit_factor': factor, 'profit_factor_status': state,
            'expectancy_krw': int(pnl.mean()) if count else 0,
            'win_rate': round(len(wins) / count * 100, 1) if count else 0.0,
            't1_rate': rate('T1_달성'), 't2_rate': rate('T2_달성'),
            'ambiguous_count': int(trades['모호봉_발생'].sum()) if count and '모호봉_발생' in trades else 0,
            'total_trades': count, 'winning_trades': len(wins), 'losing_trades': len(losses),
            'sample_sufficient': count >= min_sample}


def format_profit_factor(value):
    if pd.isna(value):
        return '산출 불가'
    if np.isinf(value):
        return '∞ (손실 거래 없음)'
    return f'{value:.2f}'


def show_backtest_quality(metrics):
    if not metrics.get('sample_sufficient', False):
        st.warning(f"청산 거래 {metrics.get('total_trades', 0)}건으로 표본이 적습니다. 30건 미만 결과로 성과를 확정하지 마세요.")
    if metrics.get('profit_factor_status') == 'no_losses':
        st.caption('손실 거래가 없어 Profit Factor의 유한한 비율을 추정할 수 없습니다.')
    st.caption('현재 선택 종목과 현재 업종 기준의 조건부 시뮬레이션입니다. 상장폐지·과거 업종 변경·일봉 체결 순서의 편향은 완전히 제거되지 않았습니다.')
    st.caption(f"종료 시 미청산 포지션 {metrics.get('open_positions', 0)}개. 최종 자산에는 평가손익이 포함되고 거래 통계에는 청산 거래만 포함됩니다.")


def calculate_position_sizing(account_capital, risk_pct, exec_multiplier, curr_price, risk_1r, atr14, avg_amount_20,
                              buy_fee=0.0005, sell_fee=0.0020, market_slippage=0.0010, remaining_account_risk=None):
    inputs = [account_capital, risk_pct, exec_multiplier, curr_price, risk_1r, atr14, avg_amount_20]
    if not np.isfinite(inputs).all() or account_capital <= 0 or avg_amount_20 <= 0:
        return {'shares': 0, 'total_invest': 0, 'weight_pct': 0.0, 'normal_risk_loss': 0,
                'worst_gap_loss': 0, 'limiting_factor': 'INVALID_SIZING_DATA', 'real_risk_pct': 0.0}
    effective_risk_pct = (risk_pct / 100.0) * exec_multiplier
    max_risk_amount = account_capital * effective_risk_pct

    if risk_1r <= 0 or curr_price <= 0 or effective_risk_pct <= 0:
        return {"shares": 0, "total_invest": 0, "weight_pct": 0.0, "normal_risk_loss": 0, "worst_gap_loss": 0, "limiting_factor": "NO_RISK_BUDGET", "real_risk_pct": 0.0}

    unit_risk = expected_stop_risk(curr_price, curr_price - risk_1r, buy_fee, sell_fee, market_slippage)
    raw_shares = int(max_risk_amount // unit_risk) if np.isfinite(unit_risk) and unit_risk > 0 else 0
    max_liq_shares = int((avg_amount_20 * 0.01) // curr_price) if avg_amount_20 > 0 else raw_shares
    max_weight_shares = int((account_capital * 0.30) // curr_price)

    if remaining_account_risk is not None:
        max_account_shares = int(max(0, remaining_account_risk) // unit_risk) if np.isfinite(remaining_account_risk) and unit_risk > 0 else 0
    else:
        max_account_shares = raw_shares
    final_shares = max(0, min(raw_shares, max_liq_shares, max_weight_shares, max_account_shares))

    if remaining_account_risk is not None and final_shares == max_account_shares:
        limiting_factor = 'ACCOUNT_RISK_CAP (계좌 합산 손절 위험)'
    elif final_shares == raw_shares:
        limiting_factor = "RISK_BUDGET (설정 손실한도)"
    elif final_shares == max_weight_shares:
        limiting_factor = "WEIGHT_CAP (30% 계좌한도)"
    else:
        limiting_factor = "LIQUIDITY_CAP (1% 거래대금)"

    total_invest = final_shares * curr_price
    weight_pct = round((total_invest / account_capital) * 100, 1) if account_capital > 0 else 0.0
    normal_risk_loss = final_shares * unit_risk
    worst_gap_loss = final_shares * expected_stop_risk(curr_price, curr_price - risk_1r - atr14, buy_fee, sell_fee, market_slippage)
    real_risk_pct = round((normal_risk_loss / account_capital) * 100, 2) if account_capital > 0 else 0.0

    return {
        "shares": final_shares, "total_invest": int(total_invest),
        "weight_pct": weight_pct, "normal_risk_loss": math.ceil(normal_risk_loss),
        "worst_gap_loss": math.ceil(worst_gap_loss),
        "limiting_factor": limiting_factor,
        "real_risk_pct": real_risk_pct
    }

# -------------------------------------------------------------
# 7. [공통 코어] 3-Track 체결 및 일중 포지션 관리 엔진
# -------------------------------------------------------------
def entry_order_conditions(decision, trading_dates):
    if decision['scored']['track'] != 'REBOUND':
        return None
    conditions = dict(decision['pattern'])
    next_index = trading_dates.searchsorted(pd.Timestamp(conditions['signal_date']), side='right')
    conditions['valid_for_date'] = (trading_dates[next_index].strftime('%Y-%m-%d')
                                    if next_index < len(trading_dates) else None)
    return conditions


def check_3track_execution(track, b_open, b_high, b_low, ma20_entry, high_20, slippage_rate=0.0010,
                           entry_conditions=None, execution_date=None):
    if not np.isfinite([b_open, b_high, b_low]).all() or b_low <= 0 or not b_low <= b_open <= b_high:
        return False, 0.0
    if track == 'REBOUND':
        if not entry_conditions or execution_date is None:
            return False, 0.0
        signal_date = pd.Timestamp(entry_conditions['signal_date'])
        order_date = pd.Timestamp(execution_date).normalize()
        if order_date <= signal_date or (order_date - signal_date).days > 7:
            return False, 0.0
        if 'valid_for_date' in entry_conditions and (entry_conditions['valid_for_date'] is None
                or order_date != pd.Timestamp(entry_conditions['valid_for_date'])):
            return False, 0.0
        trigger = entry_conditions['trigger_price']
        invalidation = entry_conditions['invalidation_price']
        ceiling = entry_conditions['max_entry_price']
        if b_open <= invalidation or b_open > ceiling:
            return False, 0.0
        if b_open >= trigger:
            price = ceil_to_tick(b_open * (1 + slippage_rate))
        elif b_high >= trigger:
            # Daily bars cannot order a pre-entry break of support and a later breakout.
            # Conservatively cancel this ambiguous stop-entry in every execution mode.
            if b_low <= invalidation:
                return False, 0.0
            price = ceil_to_tick(trigger * (1 + slippage_rate))
        else:
            return False, 0.0
        return (True, price) if price <= ceiling else (False, 0.0)
    # Legacy tracks remain available for existing callers, with a lower gap boundary.
    if b_open < ma20_entry * .97:
        return False, 0.0

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
                              market_slippage=0.0010, limit_slippage=0.0005, execution_mode="CONSERVATIVE", trail_atr_mult=1.5):
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

    if execution_mode not in {"CONSERVATIVE", "NEUTRAL", "OPTIMISTIC"}:
        raise ValueError("Unsupported execution mode")
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
        else:
            prioritize_stop = (b_close < b_open)
    else:
        prioritize_stop = hit_stop_candidate

    # 1. 오버나이트 갭다운 손절
    if h_days > 1 and b_open <= c_stop:
        exit_event = "GAP_DOWN_STOP" if not pos["t1_hit"] else "GAP_DOWN_TRAILING"
        exit_price = floor_to_tick(b_open * (1.0 - market_slippage))
    # 2. 장중 손절 터치
    elif prioritize_stop and hit_stop_candidate:
        exit_event = "STOP_LOSS" if not pos["t1_hit"] else "TRAILING_STOP"
        exit_price = floor_to_tick(c_stop * (1.0 - market_slippage))
    else:
        # 3. T1 도달 (30%)
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

            if b_low <= e_price and cur_shares > 0:
                is_ambiguous = True
            if (b_low <= e_price and cur_shares > 0
                    and (execution_mode == "CONSERVATIVE"
                         or (execution_mode == "NEUTRAL" and b_close < b_open))):
                exit_event = "T1_THEN_BREAKEVEN"
                exit_price = floor_to_tick(e_price * (1.0 - market_slippage))

        # 4. 1.5R 도달 시 +0.5R 보호 스탑 상향
        if exit_event is None and pos["t1_hit"] and not pos["t2_hit"]:
            r1 = pos["t1"] - e_price
            if b_high >= e_price + (1.5 * r1):
                pos["current_stop"] = max(pos["current_stop"], floor_to_tick(e_price + 0.5 * r1))

        # 5. T2 도달 (40%) & +1.0R 보호 스탑 상향
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

        # 6. T2 달성 후 고점 대비 trail_atr_mult ATR 다이내믹 트레일링
        if exit_event is None and pos["t2_hit"]:
            dynamic_trail = floor_to_tick(pos["peak_price"] - (trail_atr_mult * pos["atr"]))
            pos["current_stop"] = max(pos["current_stop"], dynamic_trail)

        # 7. T3 도달 (잔여 전량)
        if exit_event is None and b_high >= pos["t3"]:
            pos["t3_hit"] = True
            exit_event = "TARGET_3R_FULL"
            exit_price = floor_to_tick(pos["t3"] * (1.0 - limit_slippage))
        # 8. Closing exits occur after intraday target fills. Existing gap/stop
        # priorities and the execution-mode ambiguity assumptions stay in force.
        elif exit_event is None and pos["t1_hit"] and b_close < ma5_val and b_close < prev_low:
            exit_event = "TREND_BREAK_EXIT"
            exit_price = floor_to_tick(b_close * (1.0 - market_slippage))
        # 9. Time Stop
        elif exit_event is None and h_days >= max_holding:
            exit_event = "TIME_STOP"
            exit_price = floor_to_tick(b_close * (1.0 - market_slippage))

    pos["ambiguous_seen"] = pos.get("ambiguous_seen", False) or is_ambiguous
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
            "모호봉_발생": pos["ambiguous_seen"]
        }
        return None, cash_gain_today, trade_record

    return pos, cash_gain_today, None

# -------------------------------------------------------------
# 8. [단일 종목] 일별 MTM 모의 백테스터 (eval_start_date 워밍업 완결)
# -------------------------------------------------------------
def run_daily_mtm_backtest(df_hist, df_dual_regime, historical_supply_series=None, sector_slice=None, max_holding=10, initial_capital=30000000, risk_pct=1.0,
                           buy_cost=0.05, sell_cost=0.20, market_slippage=0.0010, limit_slippage=0.0005, execution_mode="CONSERVATIVE", min_score=70, trail_atr_mult=1.5, eval_start_date=None, _feature_cache=None):
    """
    eval_start_date 이전 종가로 첫 거래일 주문을 준비한다.
    _feature_cache는 동일한 입력·워밍업 구간의 WFO 조합에서만 공유한다.
    포지션, 수량, 체결, 현금 흐름은 각 조합에서 독립적으로 계산한다.
    """
    if len(df_hist) < 61 or not valid_price_history(df_hist, min_rows=len(df_hist)) or df_dual_regime.empty: return pd.DataFrame(), pd.DataFrame(), {}

    df_p = df_hist.copy()
    df_p.index = pd.to_datetime(df_p.index).normalize()
    df_p = df_p.loc[~df_p.index.duplicated(keep="last")].sort_index()
    df_p = prepare_engine_history(df_p)
    df_reg = df_dual_regime.copy()
    df_reg.index = pd.to_datetime(df_reg.index).normalize()
    df_reg = df_reg.loc[~df_reg.index.duplicated(keep="last")].sort_index()
    market_history = df_reg[['KS_Close']].rename(columns={'KS_Close': 'Close'})

    buy_fee = buy_cost / 100.0
    sell_fee = sell_cost / 100.0

    cash = float(initial_capital)
    active_pos = None
    pending_entry = None

    daily_records = []
    closed_trades = []

    # 워밍업 인덱스 결정
    if eval_start_date is not None:
        eval_dt = pd.Timestamp(eval_start_date).normalize()
        start_idx = max(60, df_p.index.searchsorted(eval_dt))
    else:
        start_idx = 60

    if start_idx >= len(df_p):
        return pd.DataFrame(), pd.DataFrame(), {}
    # Seed the first evaluation-day order using the preceding close only.
    for idx in range(start_idx - 1, len(df_p)):
        cur_date = df_p.index[idx]
        cur_date_str = cur_date.strftime("%Y-%m-%d")
        bar = df_p.iloc[idx]
        b_open, b_high, b_low, b_close = float(bar['Open']), float(bar['High']), float(bar['Low']), float(bar['Close'])

        # A. 3-Track 진입 실행
        if active_pos is None and pending_entry is not None:
            executed, e_price = check_3track_execution(
                pending_entry["track"], b_open, b_high, b_low,
                pending_entry["ma20_entry"], pending_entry["high_20"],
                slippage_rate=market_slippage, entry_conditions=pending_entry.get('entry_conditions'),
                execution_date=cur_date
            )
            if executed and e_price > 0:
                plan = calculate_execution_levels(df_p.iloc[:idx], e_price,
                                                  (pending_entry.get('entry_conditions') or {}).get('invalidation_price'))
                if not plan["is_excessive_stop"]:
                    clean_amt = get_clean_avg_amount(df_p.iloc[:idx], 20)
                    pos = calculate_position_sizing(cash, risk_pct, pending_entry["multiplier"], e_price, plan["risk_1r"], plan["atr14"], clean_amt, buy_fee=buy_fee, sell_fee=sell_fee, market_slippage=market_slippage)
                    shares = min(pos["shares"], int(cash // (e_price * (1.0 + buy_fee))))
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
                market_slippage=market_slippage, limit_slippage=limit_slippage, execution_mode=execution_mode,
                trail_atr_mult=trail_atr_mult
            )
            cash += cash_gain
            if trade_rec is not None:
                closed_trades.append(trade_rec)

        # C. 일별 종가 MTM 시가평가
        holding_val = (active_pos["shares"] * b_close) if active_pos is not None else 0.0
        tot_equity = cash + holding_val
        if idx >= start_idx:
            daily_records.append({
                "Date": cur_date, "Cash": cash, "Stock_Value": holding_val,
                "Total_Equity": tot_equity, "In_Position": (active_pos is not None)
            })

        # D. 장 마감 후 익일 시그널 평가
        if active_pos is None and pending_entry is None:
            if _feature_cache is not None and cur_date in _feature_cache:
                reg_code, features = _feature_cache[cur_date]
                dec = decide_execution(features, reg_code, min_score)
            else:
                reg_code = regime_at(df_reg, cur_date)
                ks_sub = market_history.loc[:cur_date]
                sec_sub = sector_slice.loc[:cur_date] if sector_slice is not None else None

                has_frgn = False
                if historical_supply_series is not None and not historical_supply_series.empty:
                    if cur_date in historical_supply_series.index:
                        has_frgn = bool(historical_supply_series.loc[cur_date] > 0)
                    else:
                        past_s = historical_supply_series.loc[historical_supply_series.index <= cur_date]
                        if not past_s.empty and (cur_date - past_s.index[-1]).days <= 7: has_frgn = bool(past_s.iloc[-1] > 0)

                if _feature_cache is None:
                    dec = evaluate_full_execution_pipeline(df_p.iloc[:idx+1], reg_code, kospi_slice=ks_sub, sector_slice=sec_sub, has_frgn=has_frgn, min_score=min_score)
                else:
                    features = evaluate_execution_features(df_p.iloc[:idx+1], reg_code, kospi_slice=ks_sub, sector_slice=sec_sub, has_frgn=has_frgn)
                    _feature_cache[cur_date] = (reg_code, features)
                    dec = decide_execution(features, reg_code, min_score)
            if "BUY" in dec["action"]:
                pending_entry = {
                    "regime": reg_code,
                    "multiplier": dec["multiplier"],
                    "track": dec["scored"]["track"],
                    "high_20": dec["scored"]["high_20"],
                    "ma20_entry": dec["scored"]["ma20"],
                    "entry_conditions": entry_order_conditions(dec, df_p.index)
                }

    if not daily_records:
        return pd.DataFrame(), pd.DataFrame(closed_trades), {}
    daily_df = pd.DataFrame(daily_records).set_index("Date")
    trades_df = pd.DataFrame(closed_trades)

    if daily_df.empty: return daily_df, trades_df, {}

    daily_df['Peak'] = daily_df['Total_Equity'].cummax().clip(lower=initial_capital)
    daily_df['Drawdown_pct'] = ((daily_df['Total_Equity'] - daily_df['Peak']) / daily_df['Peak']) * 100.0
    daily_mdd = round(abs(daily_df['Drawdown_pct'].min()), 2)
    tot_ret_pct = round(((daily_df['Total_Equity'].iloc[-1] - initial_capital) / initial_capital) * 100.0, 2)

    metrics = summarize_trades(trades_df)
    metrics.update(final_equity=int(daily_df['Total_Equity'].iloc[-1]),
                   total_return_pct=tot_ret_pct, daily_mdd=daily_mdd,
                   open_positions=int(active_pos is not None))

    return daily_df, trades_df, metrics

# -------------------------------------------------------------
# 9. [포트폴리오 바스켓] 다종목 동시 자본 배분 시뮬레이터
# -------------------------------------------------------------
def run_portfolio_backtest(ticker_list, df_dual_regime, df_krx=None, max_positions=3, max_holding=10, initial_capital=50000000, risk_pct=1.0,
                           buy_cost=0.05, sell_cost=0.20, market_slippage=0.0010, limit_slippage=0.0005, execution_mode="CONSERVATIVE", min_score=70, trail_atr_mult=1.5,
                           max_total_risk_pct=3.0, max_sector_risk_pct=2.0,
                           max_correlated_risk_pct=2.0, correlation_threshold=0.8):
    s_date = (market_today() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
    data_dict = {}
    frgn_dict = {}
    sec_dict = {}
    sector_names = {}
    excluded_symbols = []
    if min(max_total_risk_pct, max_sector_risk_pct, max_correlated_risk_pct) <= 0:
        raise ValueError('Risk limits must be positive')
    if not 0 <= correlation_threshold <= 1:
        raise ValueError('Correlation threshold must be between 0 and 1')

    for item in ticker_list:
        clean_item = str(item).strip()
        if not clean_item: continue
        code = None
        if df_krx is not None and not df_krx.empty:
            code, _ = find_stock_code(clean_item, df_krx)
        if not code and clean_item.isdigit():
            code = clean_item.zfill(6)
        if not code:
            excluded_symbols.append(clean_item)
            continue

        try:
            h = completed_live_history(load_price_history(code, s_date))
            if h is not None and len(h) >= 120 and valid_price_history(h, min_rows=len(h)):
                h.index = pd.to_datetime(h.index).normalize()
                data_dict[code] = prepare_engine_history(h)
                frgn_dict[code] = load_historical_foreign_series(code)
                # 전 종목 섹터 ETF 자동 매핑 및 시계열 수신
                sec_name = get_sector_for_stock(code, df_krx)
                sector_names[code] = sec_name
                sec_dict[code] = load_sector_etf_series(sec_name, s_date)
            else:
                excluded_symbols.append(code)
        except Exception as error:
            excluded_symbols.append(code)
            data_dict.pop(code, None)
            frgn_dict.pop(code, None)
            sec_dict.pop(code, None)
            sector_names.pop(code, None)
            logger.warning('Portfolio data unavailable [%s]: %s', code, type(error).__name__)
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
        new_entry_risk = np.nan

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
                    slippage_rate=market_slippage, entry_conditions=item.get('entry_conditions'),
                    execution_date=cur_date
                )
                if not executed or e_price <= 0: continue

                plan = calculate_execution_levels(df_code.iloc[:-1], e_price,
                                                  (item.get('entry_conditions') or {}).get('invalidation_price'))
                if plan["is_excessive_stop"]: continue

                stk_val_at_open = sum(p["shares"] * float(data_dict[c].loc[cur_date, 'Open']) for c, p in positions.items())
                equity_at_open = cash + stk_val_at_open

                clean_amt = get_clean_avg_amount(df_code.iloc[:-1], 20)
                pos_calc = calculate_position_sizing(
                    equity_at_open, risk_pct, item["multiplier"], e_price, plan["risk_1r"], plan["atr14"], clean_amt,
                    buy_fee=buy_fee, sell_fee=sell_fee, market_slippage=market_slippage
                )
                target_shares = pos_calc["shares"]

                max_cash_shares = int(cash // (e_price * (1.0 + buy_fee)))
                risk_limit = limit_portfolio_entry(
                    min(target_shares, max_cash_shares), code, sector_names[code],
                    e_price, plan['stop_price'], equity_at_open, positions, data_dict, cur_date,
                    max_total_risk_pct, max_sector_risk_pct, max_correlated_risk_pct,
                    correlation_threshold, buy_fee, sell_fee, market_slippage)
                shares = risk_limit['shares']
                req_cash = shares * e_price * (1.0 + buy_fee)

                if shares > 0 and cash >= req_cash:
                    cash -= req_cash
                    new_entry_risk = max(0 if pd.isna(new_entry_risk) else new_entry_risk,
                                         risk_limit['total_risk_pct'])
                    positions[code] = {
                        "code": code,
                        "sector": sector_names[code],
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
                market_slippage=market_slippage, limit_slippage=limit_slippage, execution_mode=execution_mode,
                trail_atr_mult=trail_atr_mult
            )
            cash += cash_gain
            if trade_rec is not None:
                closed_trades.append(trade_rec)
                closed_codes.append(code)
            else:
                positions[code] = updated_p

        for c in closed_codes: del positions[c]

        # 3. MTM 일별 시가평가
        stk_val = sum(p["shares"] * float(data_dict[c].loc[cur_date, 'Close']) for c, p in positions.items())
        daily_records.append({
            "Date": cur_date, "Cash": cash, "Stock_Value": stk_val,
            "Total_Equity": cash + stk_val, "Num_Positions": len(positions),
            "New_Entry_Stop_Risk_pct": new_entry_risk
        })

        # 4. 익일 진입 스캔 (과거 외국인 수급 및 전 종목 섹터 RS 완전 연동)
        if len(positions) < max_positions:
            reg_code = regime_at(df_dual_regime, cur_date)
            ks_sub = df_dual_regime[['KS_Close']].rename(columns={'KS_Close': 'Close'}).loc[:cur_date]

            for code in data_dict.keys():
                if code not in positions:
                    df_sub = data_dict[code].loc[:cur_date]

                    has_frgn = False
                    f_series = frgn_dict.get(code)
                    if f_series is not None and not f_series.empty:
                        past_f = f_series.loc[f_series.index <= cur_date]
                        if not past_f.empty and (cur_date - past_f.index[-1]).days <= 7: has_frgn = bool(past_f.iloc[-1] > 0)

                    sec_df = sec_dict.get(code)
                    sec_sub = sec_df.loc[:cur_date] if (sec_df is not None and not sec_df.empty) else None

                    dec = evaluate_full_execution_pipeline(df_sub, reg_code, kospi_slice=ks_sub, sector_slice=sec_sub, has_frgn=has_frgn, min_score=min_score)
                    if "BUY" in dec["action"]:
                        pending_signals.append({
                            "code": code,
                            "multiplier": dec["multiplier"],
                            "score": dec["scored"]["score"],
                            "track": dec["scored"]["track"],
                            "high_20": dec["scored"]["high_20"],
                            "ma20_entry": dec["scored"]["ma20"],
                            "entry_conditions": entry_order_conditions(dec, data_dict[code].index)
                        })

    if not daily_records:
        return pd.DataFrame(), pd.DataFrame(closed_trades), {}
    port_df = pd.DataFrame(daily_records).set_index("Date")
    p_trades = pd.DataFrame(closed_trades)
    if port_df.empty: return port_df, p_trades, {}

    port_df['Peak'] = port_df['Total_Equity'].cummax().clip(lower=initial_capital)
    port_df['Drawdown_pct'] = ((port_df['Total_Equity'] - port_df['Peak']) / port_df['Peak']) * 100.0
    p_mdd = round(abs(port_df['Drawdown_pct'].min()), 2)
    p_ret = round(((port_df['Total_Equity'].iloc[-1] - initial_capital) / initial_capital) * 100.0, 2)

    metrics = summarize_trades(p_trades)
    entries = port_df['New_Entry_Stop_Risk_pct'].dropna()
    metrics.update(final_equity=int(port_df['Total_Equity'].iloc[-1]), return_pct=p_ret,
                   daily_mdd=p_mdd, open_positions=len(positions),
                   max_entry_stop_risk_pct=float(entries.max()) if len(entries) else 0.0,
                   max_total_risk_pct=max_total_risk_pct, excluded_symbols=excluded_symbols)

    return port_df, p_trades, metrics

# -------------------------------------------------------------
# 10. [진성 3D Walk-Forward Optimization & 3-모드 삼각비교 엔진]
# -------------------------------------------------------------
def run_tricorder_backtest(df_hist, df_dual_regime, historical_supply_series=None, sector_slice=None, initial_capital=30000000, risk_pct=1.0, max_holding=10):
    results = {}
    for mode in ["CONSERVATIVE", "NEUTRAL", "OPTIMISTIC"]:
        daily_df, trades_df, met = run_daily_mtm_backtest(
            df_hist, df_dual_regime, historical_supply_series=historical_supply_series, sector_slice=sector_slice,
            initial_capital=initial_capital, risk_pct=risk_pct, execution_mode=mode, max_holding=max_holding
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
            "모호봉 발생 거래 수": f"{results['CONSERVATIVE']['ambiguous_count']}건"
        },
        {
            "체결 모드": "중립적 (Candle-Flow)",
            "최종 잔고": f"{results['NEUTRAL']['final_equity']:,}원",
            "수익률": f"{results['NEUTRAL']['total_return_pct']:+}%",
            "일별 MTM MDD": f"-{results['NEUTRAL']['daily_mdd']}%",
            "Profit Factor": results['NEUTRAL']['profit_factor'],
            "승률": f"{results['NEUTRAL']['win_rate']}%",
            "모호봉 발생 거래 수": f"{results['NEUTRAL']['ambiguous_count']}건"
        },
        {
            "체결 모드": "낙관적 (Target-First)",
            "최종 잔고": f"{results['OPTIMISTIC']['final_equity']:,}원",
            "수익률": f"{results['OPTIMISTIC']['total_return_pct']:+}%",
            "일별 MTM MDD": f"-{results['OPTIMISTIC']['daily_mdd']}%",
            "Profit Factor": results['OPTIMISTIC']['profit_factor'],
            "승률": f"{results['OPTIMISTIC']['win_rate']}%",
            "모호봉 발생 거래 수": f"{results['OPTIMISTIC']['ambiguous_count']}건"
        }
    ])
    return comp_df, results

def run_slippage_stress_test(df_hist, df_dual_regime, historical_supply_series=None, sector_slice=None, initial_capital=30000000, risk_pct=1.0):
    slippage_levels = [0.0005, 0.0010, 0.0020, 0.0030, 0.0050]
    stress_results = []
    for s_rate in slippage_levels:
        _, _, met = run_daily_mtm_backtest(
            df_hist, df_dual_regime, historical_supply_series=historical_supply_series, sector_slice=sector_slice,
            initial_capital=initial_capital, risk_pct=risk_pct,
            market_slippage=s_rate, limit_slippage=s_rate/2.0
        )
        if met:
            stress_results.append({
                "시장가 슬리피지": f"{s_rate*100:.2f}%",
                "지정가 슬리피지": f"{s_rate*50:.2f}%",
                "수익률(%)": met["total_return_pct"],
                "일별 MTM MDD(%)": met["daily_mdd"],
                "Profit Factor": format_profit_factor(met["profit_factor"]),
                "거래당 기댓값(원)": met["expectancy_krw"],
                "승률(%)": met["win_rate"]
            })
    return pd.DataFrame(stress_results)

def run_multidimensional_wfo(df_hist, df_dual_regime, historical_supply_series=None, sector_slice=None, windows=4, initial_capital=30000000, min_is_trades=20, min_oos_trades=10):
    """
    [수정] WFO 워밍업 정합성 완전 해결:
    IS/OOS 분할 시 직전 60거래일 워밍업 캔들을 사전 공급하고 eval_start_date부터 실제 매매를 평가
    """
    df_p = df_hist.copy()
    df_p.index = pd.to_datetime(df_p.index).normalize()
    df_p = df_p.loc[~df_p.index.duplicated(keep="last")].sort_index()
    df_p = prepare_engine_history(df_p)
    total_len = len(df_p)
    if total_len < 250: return pd.DataFrame()

    if windows < 1 or min_is_trades < 1 or min_oos_trades < 1:
        raise ValueError("windows and sample thresholds must be positive")
    segment_len = (total_len - 60) // (windows + 2)
    if segment_len < 20:
        return pd.DataFrame()
    wf_records = []

    grid_scores = [70, 75, 80]
    grid_trails = [1.2, 1.5, 2.0]
    grid_holds = [7, 10, 14]

    for w in range(windows):
        train_start = 60 + w * segment_len
        train_end = train_start + (segment_len * 2)
        test_end = total_len if w == windows - 1 else min(train_end + segment_len, total_len)
        if train_end >= total_len or train_end >= test_end: break

        # 워밍업 버퍼 60거래일 선행 결합
        is_warmup_start = max(0, train_start - 60)
        df_train_with_warmup = df_p.iloc[is_warmup_start:train_end]
        is_eval_start = df_p.index[train_start]

        oos_warmup_start = max(0, train_end - 60)
        df_test_with_warmup = df_p.iloc[oos_warmup_start:test_end]
        oos_eval_start = df_p.index[train_end]

        best_params = None
        best_is_fitness = -999999.0
        best_is_met = None
        default_is_result = None
        # Private to this exact training slice; never reused across windows or OOS.
        feature_cache = {}

        # 1. IS 27개 3D 조합 전수 탐색
        for sc in grid_scores:
            for tr_mult in grid_trails:
                for hd in grid_holds:
                    is_daily, _, m_tr = run_daily_mtm_backtest(
                        df_train_with_warmup, df_dual_regime, historical_supply_series=historical_supply_series, sector_slice=sector_slice,
                        initial_capital=initial_capital, min_score=sc, max_holding=hd, trail_atr_mult=tr_mult, eval_start_date=is_eval_start,
                        _feature_cache=feature_cache
                    )
                    if (sc, tr_mult, hd) == (70, 1.5, 10):
                        default_is_result = (is_daily, m_tr)
                    if m_tr and m_tr['total_trades'] >= min_is_trades and m_tr['total_return_pct'] > 0:
                        fitness = (m_tr['total_return_pct'] / max(m_tr.get('daily_mdd', 0), 1)
                                   * math.log(m_tr['total_trades'] + 1))
                        if fitness > best_is_fitness:
                            best_is_fitness = fitness
                            best_params = (sc, tr_mult, hd)
                            best_is_met = m_tr
                            best_is_days = len(is_daily)

        selection_qualified = best_params is not None
        if not best_params:
            best_params = (70, 1.5, 10)
            is_daily, best_is_met = default_is_result

            best_is_days = len(is_daily)

        # 2. OOS 3개 최적 파라미터 완전 동결 검증 (워밍업 정상화)
        opt_score, opt_trail, opt_hold = best_params
        oos_daily, _, m_te = run_daily_mtm_backtest(
            df_test_with_warmup, df_dual_regime, historical_supply_series=historical_supply_series, sector_slice=sector_slice,
            initial_capital=initial_capital, min_score=opt_score, max_holding=opt_hold, trail_atr_mult=opt_trail, eval_start_date=oos_eval_start
        )

        if m_te and best_is_met:
            is_ret = best_is_met["total_return_pct"]
            oos_ret = m_te["total_return_pct"]
            is_daily_ret = (1 + is_ret / 100) ** (1 / max(best_is_days, 1)) - 1
            oos_daily_ret = (1 + oos_ret / 100) ** (1 / max(len(oos_daily), 1)) - 1
            wfe = round(oos_daily_ret / is_daily_ret * 100, 1) if is_daily_ret > 0 else None

            enough = best_is_met['total_trades'] >= min_is_trades and m_te['total_trades'] >= min_oos_trades
            finite_pf = np.isfinite(m_te['profit_factor'])
            if not enough:
                verdict = '표본 부족'
            elif not selection_qualified:
                verdict = 'IS 기준 미달 · 기본 설정 평가'
            elif not finite_pf:
                verdict = 'PF 추정 보류'
            elif wfe is not None and wfe >= 50 and m_te['profit_factor'] >= 1.1 and oos_ret > 0:
                verdict = '기준 충족 (수익 보장 아님)'
            else:
                verdict = '기준 미달'
            wf_records.append({
                "구간": f"Window {w+1}",
                "IS 구간": f"{is_eval_start.strftime('%y.%m')} ~ {df_p.index[train_end-1].strftime('%y.%m')}",
                "IS 최적 3D 파라미터": f"Score {opt_score}점 / Trail {opt_trail}x / Hold {opt_hold}일",
                "IS 수익률(%)": is_ret,
                "IS PF": format_profit_factor(best_is_met["profit_factor"]),
                "IS 청산 거래": best_is_met["total_trades"],
                "OOS 블라인드 구간": f"{oos_eval_start.strftime('%y.%m')} ~ {df_p.index[test_end-1].strftime('%y.%m')}",
                "OOS 수익률(%)": oos_ret,
                "OOS PF": format_profit_factor(m_te["profit_factor"]),
                "OOS 청산 거래": m_te["total_trades"],
                "WFE(%)": f"{wfe}%" if wfe is not None else "산출 불가 (IS 수익률 ≤ 0)",
                "검증 기준 판정": verdict
            })

    return pd.DataFrame(wf_records)

# -------------------------------------------------------------
# [UI 라우팅 및 뷰 구성]
# -------------------------------------------------------------
def invalidate_scan_results():
    for key in ('screened_res', 'screened_watch', 'screened_meta', 'scan_stock_choice'):
        st.session_state.pop(key, None)


def refresh_market_data():
    st.cache_data.clear()
    invalidate_scan_results()


def scan_result_context(regime, min_cap, scan_pool, include_supply):
    return (str(market_today()), regime.get('data_date'), regime['code'],
            regime['multiplier'], min_cap, scan_pool, include_supply, 'SUPPORT-REBOUND-1')


def scan_results_are_current(metadata, context):
    if not metadata or metadata.get('context') != context:
        return False
    age = datetime.datetime.now(datetime.timezone.utc).timestamp() - metadata.get('created_at', 0)
    return 0 <= age < 3600


def transfer_scanned_stock():
    metadata = st.session_state.get('screened_meta')
    if not scan_results_are_current(metadata, metadata.get('context') if metadata else None):
        invalidate_scan_results()
        return
    selected = st.session_state['scan_stock_choice']
    st.session_state['selected_stock'] = selected
    for key in ('diagnosis_stock', 'backtest_stock', 'stress_stock'):
        st.session_state[key] = selected


st.sidebar.button('시세·시장 데이터 새로 조회', on_click=refresh_market_data)

dual_regime = get_current_dual_regime()

st.title("⚖️ QUANT-EXECUTION ENGINE PRO MAX")
st.caption('전략 버전: SUPPORT-REBOUND-1 · MA20 지지·반등 확인 후 다음 거래일 조건부 진입')
st.caption("“주가를 예측하지 않고, **틀렸을 때 손실을 제한하고 맞았을 때 분할 익절**하도록 통제하는 기계적 실행 시스템”")

c_rg1, c_rg2 = st.columns([1, 3])
with c_rg1:
    st.metric("최근 데이터 기준 듀얼 시장 국면", dual_regime["status"], f"배분 배수: {dual_regime['multiplier']}x")
with c_rg2:
    (st.warning if dual_regime["code"] == "UNKNOWN" else st.info)(f"💡 **시장 대응 지침**: {dual_regime['msg']}")
st.caption(f"시장 데이터 기준일: {dual_regime.get('data_date') or '확인 불가'} · 시세 조회 캐시 최대 1시간. 장중 데이터는 미완성 일봉일 수 있습니다.")

if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = "삼성전자"
for stock_key in ('diagnosis_stock', 'backtest_stock', 'stress_stock'):
    st.session_state.setdefault(stock_key, st.session_state['selected_stock'])

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎯 STEP 1 · 실행 중심 퀀트 진단 & 포지션 계산",
    "📡 STEP 2 · 코스피 Buy Zone 스크리너",
    "🧪 STEP 3 · 단일 종목 실계좌 MTM 백테스트",
    "💼 STEP 4 · 다종목 포트폴리오 시뮬레이터",
    "🔬 STEP 5 · 3D Walk-Forward(WFO) & 스트레스 테스트",
    "📘 주린이 눈높이 실전 매뉴얼",
    "📖 시스템 무결성 헌장"
])

# -------------------------------------------------------------
# TAB 1: 퀀트 진단 & 포지션 계산
# -------------------------------------------------------------
with tab1:
    st.subheader("정밀 실행 계획 & 리스크 기반 포지션 산출")

    with st.expander("💡 [주린이 30초 요약] 이 화면을 어떻게 읽고 행동해야 하나요? (클릭하여 열기)"):
        st.markdown("""
        * **1. 최종 판정 확인**: **조건부 진입**일 때만 다음 거래일 트리거 주문을 검토합니다. 관찰·시장 차단 상태에서는 주문하지 않습니다.
        * **2. 가격 4총사 메모**: HTS/MTS를 켜고 **동적 손절선**과 **1차 목표가(+1R)**를 먼저 입력합니다.
        * **3. 딱 '권장 매수 수량'만큼만 주문**: 내 원금 전체를 몰빵하지 말고, 시스템이 설정 손실률과 거래비용으로 정해준 주수만큼만 삽니다.
        * **4. 체결 즉시 스탑로스 설정**: 주식을 사자마자 증권사 앱에서 **손절가에 '자동 감시 매도'**를 걸어둡니다.
        """)

    col1, col2, col3 = st.columns([2, 1.5, 1])
    with col1:
        target_stock = st.text_input("종목명 또는 6자리 코드", key="diagnosis_stock")
    with col2:
        acc_cap = st.number_input("내 총 계좌 잔고 (원)", min_value=1000000, value=30000000, step=5000000)
    with col3:
        risk_pct_input = st.slider("1회 허용 손실률 (%)", 0.3, 2.0, 1.0, step=0.1)

    with st.expander('기존 보유분을 포함한 계좌 손절 위험 예산'):
        live_total_risk_pct = st.number_input('진단 계좌 전체 위험 한도 (%)', min_value=0.5, max_value=10.0, value=3.0, step=0.5)
        existing_stop_risk = st.number_input('기존 보유분의 예상 손절 위험 (원)', min_value=0, value=0, step=10000)
        st.caption('증권 계좌를 자동 조회하지 않습니다. 기존 보유가 있으면 거래비용을 포함한 예상 손절 손실 합계를 입력하세요. 섹터·상관 위험 한도는 STEP 4 모의 포트폴리오에 적용됩니다.')

    if st.button("실행 계획 정밀 진단", type="primary"):
        with st.spinner("호가 검증 및 100점 파이프라인 연산 중..."):
            df_krx = load_krx_listing()
            code, name = find_stock_code(target_stock, df_krx)

            if not code:
                st.error("종목을 찾을 수 없습니다.")
            else:
                s_date = (market_today() - datetime.timedelta(days=350)).strftime("%Y-%m-%d")
                df_p = completed_live_history(load_price_history(code, s_date))
                df_ks = load_price_history('KS11', s_date)

                sec_name = get_sector_for_stock(code, df_krx)
                sec_df = load_sector_etf_series(sec_name, s_date)

                if df_p is None or len(df_p) < 61:
                    st.error("지지·반등 진단에 필요한 최소 61거래일의 시세가 부족합니다.")
                else:
                    supply = get_supply_data(code, df_p.index[-1])
                    if supply["status"] != "OK":
                        st.caption("외국인 수급을 확인하지 못해 수급 가점을 적용하지 않습니다.")
                    verdict, df_p = evaluate_live_execution(
                        df_p, dual_regime, df_ks, sec_df,
                        has_frgn=(supply["status"] == "OK" and supply["frgn_5"] is not None and supply["frgn_5"] > 0)
                    )
                    pattern = verdict['pattern']
                    st.markdown(f"**패턴 상태: {pattern['label']}**")
                    st.caption(pattern['reason'])
                    if verdict['scored']['grade'] == 'UNVERIFIED':
                        st.warning(verdict['reason'])
                    else:
                        st.caption(f"종목 데이터 기준일: {df_p.index[-1].strftime('%Y-%m-%d')}")
                    if verdict['scored']['grade'] != 'UNVERIFIED':
                        levels = verdict["levels"]
                        scored = verdict["scored"]
                        curr_price = pattern.get('max_entry_price', levels['entry_price'])

                        clean_amt = get_clean_avg_amount(df_p, 20)
                        pos = calculate_position_sizing(
                            acc_cap, risk_pct_input, verdict["multiplier"],
                            curr_price, curr_price - levels["stop_price"], levels["atr14"], clean_amt,
                            remaining_account_risk=max(0, acc_cap * live_total_risk_pct / 100 - existing_stop_risk)
                        )

                        st.markdown("---")
                        if "BUY" in verdict["action"] and pos['shares'] == 0:
                            st.warning('종목 조건을 통과했지만 위험 예산·비중·유동성 한도로 매수 가능한 수량이 0주입니다.')
                        elif "BUY" in verdict["action"]:
                            st.success(f"### {verdict['badge']}\n**진입 전략 트랙**: `{scored['track']}` | **업종 매핑**: `{sec_name}` | **지침**: {verdict['reason']}")
                        elif verdict["action"] == "WAIT":
                            st.warning(f"### {verdict['badge']}\n**업종 매핑**: `{sec_name}` | **지침**: {verdict['reason']}")
                        else:
                            st.error(f"### {verdict['badge']}\n**지침**: {verdict['reason']}")

                        st.markdown("#### 💯 100점 만점 퀀트 평가 내역 (전 종목 Sector RS 자동 결합)")
                        dt_cols = st.columns(6)
                        for i, (k, v) in enumerate(scored["details"].items()):
                            dt_cols[i].metric(k, v)

                        if levels["resistance_warning"]:
                            st.warning(f"⚠️ **저항선 주의**: {levels['resistance_warning']}")

                        if pattern['state'] == 'CONFIRMED':
                            st.markdown("### 🛑 다음 거래일 조건부 진입 계획")
                            st.caption(f"트리거 {pattern['trigger_price']:,}원 · 최대 진입가 {pattern['max_entry_price']:,}원 · "
                                       f"지지 무효화 {pattern['invalidation_price']:,}원. 다음 거래일 미돌파·무효화·최대가 초과 시 취소합니다.")
                            st.caption('가격선은 트리거 체결을 가정합니다. 수량은 최대 진입가와 표시 손절가 기준으로 보수적으로 산정하며, 실제 체결 후 손절·목표가·수량을 다시 계산해야 합니다. 예상 거래비용을 포함합니다.')
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("조건부 진입 트리거", f"{levels['entry_price']:,}원", f"Wilder ATR: {levels['atr14']:,.2f}원")
                            m2.metric("동적 손절선 (보수적)", f"{levels['stop_price']:,}원", f"{levels['loss_pct']}%", delta_color="inverse")
                            m3.metric("1차 익절가 (30% 매도 & 본전스탑)", f"{levels['target_1r']:,}원", "+1.0R")
                            m4.metric("2차 익절가 (40% 매도)", f"{levels['target_2r']:,}원", "+2.0R")

                            st.markdown("### 💼 계좌 리스크 기반 포지션 및 2-Track 손실 시뮬레이션")
                            p1, p2, p3, p4 = st.columns(4)
                            p1.metric("최종 권장 매수 수량", f"{pos['shares']:,}주", f"계좌 비중: {pos['weight_pct']}%")
                            p2.metric("수량 제한 요인", pos["limiting_factor"], f"실질 리스크: {pos['real_risk_pct']}%")
                            p3.metric("비용 포함 예상 손절 손실", f"-{pos['normal_risk_loss']:,}원", "1R 원칙 통제")
                            p4.metric("스트레스 갭다운(-1ATR)", f"-{pos['worst_gap_loss']:,}원", "오버나이트 급락 가정", delta_color="inverse")

# -------------------------------------------------------------
# TAB 2: 코스피 Buy Zone 스크리너 (수급 정합성 옵션 추가)
# -------------------------------------------------------------
with tab2:
    st.subheader("코스피 Buy Zone 지지 및 반등 트리거 탐색기")
    st.caption('각 봉 직전 확정 MA20·ATR로 최근 3개 봉의 지지 시험과 회복을 확인합니다. 반등 확인·70점·시장·손절폭 조건을 통과하면 다음 거래일 고가 돌파를 기다립니다. 관찰 후보는 매수 추천이 아닙니다.')
    st.caption('당일 봉은 15:40 KST 이후 조회된 시세부터 사용합니다. 업종은 키워드 기반 추정이며 미분류는 대표지수 ETF를 사용합니다.')

    c_s1, c_s2, c_s3 = st.columns([1.5, 1.5, 1])
    with c_s1:
        min_cap = st.number_input("시가총액 하한 (억원)", 500, 20000, 2000, step=500)
    with c_s2:
        scan_pool = st.slider("스캔 풀 (거래대금 상위)", 50, 300, 100, step=50)
    with c_s3:
        include_supply_in_scan = st.checkbox("외국인 수급 5점 결합 (정밀 스캔)", value=False,
                                             help='수급 가점은 패턴 확인·시장 차단·진입 무효화 조건을 대체하지 않습니다.')

    scan_context = scan_result_context(dual_regime, min_cap, scan_pool, include_supply_in_scan)
    if "screened_res" in st.session_state and not scan_results_are_current(
            st.session_state.get('screened_meta'), scan_context):
        invalidate_scan_results()
        st.info('조회 조건·시장 상태가 변경되었거나 결과가 만료되었습니다. 다시 스캔하세요.')
    if dual_regime['code'] in {'RISK-OFF', 'UNKNOWN'}:
        st.warning('현재 시장 국면으로 신규 진입을 차단합니다. 스캔은 지지·반등 관찰 후보를 계속 보여줍니다.')

    if st.button("Buy Zone 종목 스캔 시작", type="primary"):
        invalidate_scan_results()
        with st.spinner('지지·회복·반등 상태와 조건부 진입 요건을 검증 중...'):
            df_k = load_stock_listing('KOSPI')
            df_krx = load_krx_listing()
            if df_k is None or df_k.empty:
                st.error('코스피 종목 목록 조회 실패: 후보군을 만들 수 없어 스캔을 완료하지 못했습니다.')
            elif not {'Code', 'Name', 'Close', 'Volume', 'Marcap'}.issubset(df_k.columns):
                st.error('코스피 종목 목록의 필수 항목이 누락되어 스캔을 중단했습니다.')
            else:
                df_k = df_k.copy()
                df_k['Code'] = df_k['Code'].astype(str).str.zfill(6)
                for num_col in ['Close', 'Volume', 'Amount', 'Marcap']:
                    if num_col in df_k.columns:
                        df_k[num_col] = pd.to_numeric(df_k[num_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
                if 'Amount' not in df_k.columns or (df_k['Amount'] == 0).all():
                    df_k['Amount'] = df_k['Close'] * df_k['Volume']
                eligible_pool = df_k[df_k['Marcap'] >= min_cap * 100000000]
                pool = eligible_pool.sort_values(by='Amount', ascending=False).head(scan_pool)
                screened, watch = [], []
                sector_lookup = build_sector_lookup(df_krx) if not pool.empty else {}
                scan_counts = {'evaluated': 0, 'missing': 0, 'failed': 0, 'supply_missing': 0}
                scan_errors, pattern_counts, gate_counts = {}, {}, {}
                p_bar = st.progress(0)
                s_date = (market_today() - datetime.timedelta(days=180)).isoformat()
                df_ks = load_price_history('KS11', s_date)
                for idx, (_, r) in enumerate(pool.iterrows()):
                    p_bar.progress((idx + 1) / len(pool))
                    code, name = str(r['Code']).zfill(6), r['Name']
                    try:
                        h = completed_live_history(load_price_history(code, s_date))
                        if not valid_price_history(h, min_rows=61):
                            scan_counts['missing'] += 1
                            gate_counts['DATA'] = gate_counts.get('DATA', 0) + 1
                            continue
                        sec_name = get_sector_for_stock(code, df_krx, sector_lookup=sector_lookup)
                        sec_df = load_sector_etf_series(sec_name, s_date)
                        has_frgn = False
                        if include_supply_in_scan:
                            sup = get_supply_data(code, h.index[-1])
                            has_frgn = sup['status'] == 'OK' and sup['frgn_5'] is not None and sup['frgn_5'] > 0
                            if sup['status'] != 'OK':
                                scan_counts['supply_missing'] += 1
                        dec, h = evaluate_live_execution(h, dual_regime, df_ks, sec_df, has_frgn)
                        pattern, levels = dec['pattern'], dec['levels']
                        gate = dec['block_code']
                        gate_counts[gate] = gate_counts.get(gate, 0) + 1
                        pattern_counts[pattern['label']] = pattern_counts.get(pattern['label'], 0) + 1
                        verified = dec['scored']['grade'] != 'UNVERIFIED'
                        scan_counts['evaluated' if verified else 'missing'] += 1
                        if pattern['state'] not in {'APPROACH', 'RECOVERED', 'CONFIRMED'}:
                            continue
                        row = {
                            '코드': code, '종목명': name, '현재가': int(h['Close'].iloc[-1]),
                            '시세기준일': h.index[-1].strftime('%Y-%m-%d'), '패턴상태': pattern['label'],
                            '스코어': dec['scored']['score'] if verified else None,
                            '매핑섹터': sec_name, '최종판정': dec['badge'], '보류·진입 근거': dec['reason'],
                            '지지선': pattern.get('support'),
                            '진입트리거': pattern.get('trigger_price') if verified else None,
                            '최대진입가': pattern.get('max_entry_price') if verified else None,
                            '무효화가': pattern.get('invalidation_price') if verified else None,
                            '손절가': levels['stop_price'] if pattern['state'] == 'CONFIRMED' and verified else None,
                            '손절폭(%)': levels['loss_pct'] if pattern['state'] == 'CONFIRMED' and verified else None,
                            '거래대금(억)': int(r['Amount'] / 100000000),
                        }
                        (screened if dec['action'] in {'BUY', 'BUY_REDUCED'} else watch).append(row)
                    except Exception as error:
                        scan_counts['failed'] += 1
                        error_type = type(error).__name__
                        scan_errors[error_type] = scan_errors.get(error_type, 0) + 1
                        logger.warning('Scanner evaluation failed [%s]: %s', code, error_type)
                p_bar.empty()
                st.session_state['screened_res'] = pd.DataFrame(screened)
                st.session_state['screened_watch'] = pd.DataFrame(watch)
                st.session_state['screened_meta'] = {
                    'context': scan_context, 'created_at': datetime.datetime.now(datetime.timezone.utc).timestamp(),
                    'counts': scan_counts, 'errors': scan_errors, 'patterns': pattern_counts, 'gates': gate_counts,
                    'listed': len(df_k), 'eligible': len(eligible_pool), 'scanned': len(pool),
                }
                if screened:
                    st.success(f'조건부 진입 {len(screened)}개 · 관찰 후보 {len(watch)}개. 진입은 다음 거래일 트리거 충족 시에만 가능합니다.')
                elif dual_regime['code'] in {'RISK-OFF', 'UNKNOWN'}:
                    st.warning(f"시장 {dual_regime['code']}로 신규 진입 0개 · 관찰 후보 {len(watch)}개입니다.")
                elif not len(pool):
                    st.warning('시가총액 조건을 충족하는 스캔 후보가 없습니다.')
                elif not scan_counts['evaluated']:
                    st.warning('데이터 미확인으로 평가를 완료하지 못했습니다. 단계별 내역을 확인하세요.')
                else:
                    st.info(f'조건부 진입 0개 · 관찰 후보 {len(watch)}개. 패턴·점수·손절폭별 보류 사유를 확인하세요.')

    if 'screened_meta' in st.session_state:
        scan_meta = st.session_state['screened_meta']
        counts = scan_meta['counts']
        scanned_at = datetime.datetime.fromtimestamp(scan_meta['created_at'], datetime.timezone(datetime.timedelta(hours=9)))
        st.caption(f"스캔 완료: {scanned_at:%Y-%m-%d %H:%M:%S} KST · 시장 기준일: {dual_regime.get('data_date') or '확인 불가'} · "
                   f"평가 완료 {counts['evaluated']}개 / 데이터 미확인 {counts['missing']}개 / 처리 실패 {counts['failed']}개")
        if counts['supply_missing']:
            st.caption(f"외국인 수급 미확인 {counts['supply_missing']}개: 수급 가점 없이 평가했습니다.")
        if scan_meta['errors']:
            st.warning('일부 종목 처리 실패: ' + ', '.join(f'{name} {count}건' for name, count in scan_meta['errors'].items()))
        if 'patterns' in scan_meta:
            with st.expander('스캔 단계별 판정 내역'):
                st.caption(f"전체 목록 {scan_meta['listed']}개 → 시가총액 통과 {scan_meta['eligible']}개 → 거래대금 순 스캔 {scan_meta['scanned']}개")
                st.write('패턴 상태별 종목 수', scan_meta['patterns'])
                gate_labels = {'DATA': '데이터 미확인', 'MARKET': '시장 차단', 'PATTERN': '반등 미확인',
                               'SCORE': '점수 미달', 'STOP': '손절폭 초과', 'ELIGIBLE': '조건부 진입 가능'}
                st.write('최종 보류·허용 사유 (종목별 첫 차단 조건)',
                         {gate_labels[k]: v for k, v in scan_meta['gates'].items()})

    candidates = []
    if 'screened_res' in st.session_state and not st.session_state['screened_res'].empty:
        st.markdown('#### 조건부 진입 후보 · 다음 거래일 트리거 대기')
        df_show = st.session_state['screened_res'].sort_values('스코어', ascending=False) if '스코어' in st.session_state['screened_res'] else st.session_state['screened_res']
        st.dataframe(df_show, width='stretch')
        candidates.extend(df_show['종목명'].tolist())
    if 'screened_watch' in st.session_state and not st.session_state['screened_watch'].empty:
        st.markdown('#### 관찰 후보 · 현재 신규 진입 보류')
        df_watch = st.session_state['screened_watch'].sort_values('스코어', ascending=False, na_position='last')
        st.dataframe(df_watch, width='stretch')
        candidates.extend(df_watch['종목명'].tolist())
    if candidates:
        st.selectbox('진단 탭으로 전송할 종목 선택', list(dict.fromkeys(candidates)), key='scan_stock_choice')
        if st.button('선택 종목을 STEP 1으로 전송', on_click=transfer_scanned_stock):
            st.info(f"'{st.session_state['selected_stock']}' 선택 완료. 진단·백테스트·스트레스 검증 탭에 반영했습니다.")

# -------------------------------------------------------------
# TAB 3: 단일 종목 실계좌 MTM 백테스트 & 3-모드 삼각비교
# -------------------------------------------------------------
with tab3:
    st.subheader("단일 종목 일별 시가평가(Daily MTM) 모의 백테스트")
    st.caption("일봉 체결 후 고저 순서는 알 수 없어 체결 모드별 가정을 적용합니다. REBOUND 조건부 진입, 부분익절 누적 현금흐름 완결 회계, 과거 5일 외국인 수급 및 전 종목 자동 Sector RS를 동시 반영합니다.")

    bt_stock = st.text_input("백테스트 대상 종목명 또는 코드", key="backtest_stock")
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
        with st.spinner(f"'{bt_stock}' 과거 2년 수급·섹터 시계열 동기화 및 MTM 연산 중..."):
            df_krx = load_krx_listing()
            b_code, b_name = find_stock_code(bt_stock, df_krx)
            if not b_code:
                st.error("종목을 찾을 수 없습니다.")
            else:
                s_date = (market_today() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
                df_bt = completed_live_history(load_price_history(b_code, s_date))
                dual_hist = completed_live_history(load_historical_dual_regime(s_date))
                frgn_hist = load_historical_foreign_series(b_code)
                sec_name = get_sector_for_stock(b_code, df_krx)
                sec_df = load_sector_etf_series(sec_name, s_date)

                if start_tricorder_bt:
                    comp_df, tri_res = run_tricorder_backtest(
                        df_bt, dual_hist, historical_supply_series=frgn_hist, sector_slice=sec_df,
                        initial_capital=bt_init_cap, risk_pct=bt_risk_pct, max_holding=bt_hold
                    )
                    st.markdown("### ⚡ 3-모드 체결 불확실성 삼각비교 (Tricorder Comparison)")
                    st.dataframe(comp_df, width='stretch')
                    if comp_df.empty:
                        st.warning('데이터가 부족하여 체결 모드 비교를 실행하지 못했습니다.')
                    else:
                        show_backtest_quality(tri_res['CONSERVATIVE'])
                        st.line_chart(pd.DataFrame({
                            "보수적 (Worst-Case)": tri_res["CONSERVATIVE_daily"]["Total_Equity"],
                            "중립적 (Candle-Flow)": tri_res["NEUTRAL_daily"]["Total_Equity"],
                            "낙관적 (Target-First)": tri_res["OPTIMISTIC_daily"]["Total_Equity"]
                        }))
                else:
                    daily_df, trades_df, met = run_daily_mtm_backtest(
                        df_bt, dual_hist, historical_supply_series=frgn_hist, sector_slice=sec_df, max_holding=bt_hold,
                        initial_capital=bt_init_cap, risk_pct=bt_risk_pct,
                        execution_mode=exec_mode
                    )

                    if daily_df.empty:
                        st.warning("유효한 데이터나 평가 구간이 부족하여 백테스트를 실행하지 못했습니다.")
                    else:
                        show_backtest_quality(met)
                        st.markdown(f"### 📊 '{b_name}'({b_code}) 일별 MTM 모의 성과 요약 (섹터: {sec_name})")
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("최종 계좌 잔고", f"{met['final_equity']:,}원", f"{met['total_return_pct']:+}%")
                        k2.metric("일별 종가 MTM MDD", f"-{met['daily_mdd']}%", "Close-to-Close", delta_color="inverse")
                        k3.metric("Profit Factor (원화 기준)", format_profit_factor(met["profit_factor"]))
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
# TAB 4: 다종목 포트폴리오 시뮬레이터 (전 종목 수급·섹터·트레일 완전 일치)
# -------------------------------------------------------------
with tab4:
    st.subheader("다종목 동시 보유 포트폴리오 시뮬레이터 (Portfolio MTM)")
    st.caption("외국인 수급은 조회 가능한 과거 구간에서만 반영합니다. 섹터는 현재 업종 기준 추정이며 과거 업종 변경은 반영하지 않습니다. 전체 계좌 자산 기준 리스크 사이징, 종목별 과거 수급 5일 합계 및 전 종목 자동 Sector RS·다이내믹 Trailing을 완전 일치 검증합니다.")

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

    with st.expander('계좌 전체·섹터·상관 종목 손절 위험 한도'):
        total_risk_limit = st.number_input('계좌 전체 위험 한도 (%)', min_value=0.5, max_value=10.0, value=3.0, step=0.5)
        sector_risk_limit = st.number_input('동일 섹터 위험 한도 (%)', min_value=0.5, max_value=10.0, value=2.0, step=0.5)
        correlated_risk_limit = st.number_input('상관 종목군 위험 한도 (%)', min_value=0.5, max_value=10.0, value=2.0, step=0.5)
        st.caption('신규 매수 시 비용 포함 예상 손절 위험을 제한합니다. 상관계수는 전일까지 최대 60거래일 수익률로 계산하며 0.8 이상을 같은 위험군으로 봅니다. 추정 불가 시 보수적으로 합산합니다. 갭 손실과 이후 평가액 변동은 한도를 초과할 수 있습니다.')

    if st.button("포트폴리오 백테스트 실행", type="primary"):
        with st.spinner("다종목 전체자산 기준 사이징 & 지지·반등 조건부 진입 시뮬레이션 중..."):
            tickers = [t.strip() for t in basket_input.split(",") if t.strip()]
            df_krx = load_krx_listing()
            s_date = (market_today() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
            dual_hist = completed_live_history(load_historical_dual_regime(s_date))

            p_df, p_trades, p_met = run_portfolio_backtest(
                tickers, dual_hist, df_krx=df_krx, max_positions=max_slots,
                initial_capital=port_capital, risk_pct=port_risk, execution_mode=p_exec_mode,
                max_total_risk_pct=total_risk_limit, max_sector_risk_pct=sector_risk_limit,
                max_correlated_risk_pct=correlated_risk_limit
            )

            if p_df.empty:
                st.warning("유효한 종목·공통 거래일·시장 데이터가 부족하여 포트폴리오 평가를 실행하지 못했습니다.")
            else:
                show_backtest_quality(p_met)
                if p_met['excluded_symbols']:
                    st.warning('데이터 부족으로 제외된 종목: ' + ', '.join(p_met['excluded_symbols']))
                st.caption(f"신규 진입 후 최대 예상 손절 위험: {p_met['max_entry_stop_risk_pct']:.2f}% · 전체 한도 {p_met['max_total_risk_pct']:.2f}%")
                st.markdown("### 📊 다종목 포트폴리오 운용 성과")
                pk1, pk2, pk3, pk4 = st.columns(4)
                pk1.metric("포트폴리오 최종 잔고", f"{p_met['final_equity']:,}원", f"{p_met['return_pct']:+}%")
                pk2.metric("일별 종가 MTM MDD", f"-{p_met['daily_mdd']}%", "Close-to-Close", delta_color="inverse")
                pk3.metric("Profit Factor", format_profit_factor(p_met["profit_factor"]))
                pk4.metric("거래당 기댓값", f"{p_met['expectancy_krw']:,}원")

                st.line_chart(p_df['Total_Equity'])
                st.markdown("#### 📋 포트폴리오 체결 내역 (동일 엔진 완결)")
                st.dataframe(p_trades, width='stretch')

# -------------------------------------------------------------
# TAB 5: 진성 3D Walk-Forward WFO & 슬리피지 스트레스 테스트
# -------------------------------------------------------------
with tab5:
    st.subheader("🔬 진성 3D Walk-Forward Optimization (WFO) & 슬리피지 부하 검증")
    st.caption("IS에서 [Score x Trail x Hold] 27개 조합을 탐색하고 OOS에서 고정 평가합니다. 60거래일 워밍업과 일별 복리 수익률 기준 WFE를 사용합니다. IS 최소 20건·OOS 최소 10건의 청산 거래가 필요합니다.")

    test_stock = st.text_input("검증 대상 종목명 또는 코드", key="stress_stock")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("슬리피지 민감도 테스트 가동", type="primary"):
            with st.spinner("슬리피지 5단계 부하 시뮬레이션 중..."):
                df_krx = load_krx_listing()
                b_code, _ = find_stock_code(test_stock, df_krx)
                if b_code:
                    s_date = (market_today() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
                    df_bt = completed_live_history(load_price_history(b_code, s_date))
                    dual_hist = completed_live_history(load_historical_dual_regime(s_date))
                    frgn_hist = load_historical_foreign_series(b_code)
                    sec_name = get_sector_for_stock(b_code, df_krx)
                    sec_df = load_sector_etf_series(sec_name, s_date)

                    stress_df = run_slippage_stress_test(df_bt, dual_hist, historical_supply_series=frgn_hist, sector_slice=sec_df)
                    st.markdown("#### 📊 슬리피지 단계별 성과 변동표")
                    st.dataframe(stress_df, width='stretch')
                else:
                    st.error("종목을 찾을 수 없습니다.")

    with col_t2:
        if st.button("진성 3D Walk-Forward(WFO) 분석 가동", type="primary"):
            with st.spinner("4분할 윈도우 [Score x Trail x Hold] 27개 조합 탐색 -> OOS 블라인드 평가 중 (워밍업 정합성 보장)..."):
                df_krx = load_krx_listing()
                b_code, _ = find_stock_code(test_stock, df_krx)
                if b_code:
                    s_date = (market_today() - datetime.timedelta(days=730)).strftime("%Y-%m-%d")
                    df_bt = completed_live_history(load_price_history(b_code, s_date))
                    dual_hist = completed_live_history(load_historical_dual_regime(s_date))
                    frgn_hist = load_historical_foreign_series(b_code)
                    sec_name = get_sector_for_stock(b_code, df_krx)
                    sec_df = load_sector_etf_series(sec_name, s_date)

                    wf_df = run_multidimensional_wfo(df_bt, dual_hist, historical_supply_series=frgn_hist, sector_slice=sec_df)
                    st.markdown("#### 📊 Walk-Forward Efficiency (WFE) 결과")
                    st.dataframe(wf_df, width='stretch')
                else:
                    st.error("종목을 찾을 수 없습니다.")

# -------------------------------------------------------------
# TAB 6: 주린이 눈높이 실전 매뉴얼
# -------------------------------------------------------------
with tab6:
    st.subheader("📘 주린이 전용 실전 매뉴얼: 이 화면을 어떻게 읽고 행동할까?")
    st.caption("주가를 예측하지 않고, **언제 사고 / 어디서 도망치고 / 어디서 돈을 챙길지** 통제하는 자동 안전벨트 설명서입니다.")

    st.markdown(r"""
    ---
    ### 1. 장 시작 전 필수 확인: 시장 신호등 (Market Regime)
    화면 맨 위에 항상 떠 있는 배너는 **“지금 주식을 새로 사도 되는 날씨인가?”**를 알려주는 신호등입니다.

    | 신호등 상태 | 시장 상태 설명 | 내가 취해야 할 행동 |
    |---|---|---|
    | **🟢 FAVORABLE (우호)** | 코스피/코스닥 둘 다 20일선 위에서 상승 중 | 추천된 권장 수량(100%) 매수 가능 |
    | **🟡 NEUTRAL (중립/경계)** | 지수 하나가 흔들리거나 횡보 중 | 권장 매수 수량의 **절반(50%)만 매수** |
    | **⚪ UNKNOWN (미확인)** | 필수 시장 데이터가 없거나 오래됨 | 신규 매수 보류 |
    | **🔴 RISK-OFF (위험 회피)** | 주요 지수가 20일선 아래로 꺾인 하락장 | **신규 매수 전면 금지** (100점이어도 절대 매수 금지) |

    ---
    ### 2. STEP 1 진단 결과: 가격 4총사와 주문 수량 해석법

    #### ① 진입 전략 트랙 (어떻게 살 것인가?)
    * **REBOUND (지지 회복 후 돌파 대기)**: 전일 MA20 지지 구간의 최근 3개 봉 시험, 종가 회복, 양봉·전일 대비 상승·종가 위치·거래량을 확인합니다.
    * **조건부 진입 후보**: 반등 확인과 70점·시장·손절폭 조건을 통과한 상태입니다. 다음 거래일 반등 봉 고가 위 트리거를 돌파할 때만 진입합니다.
    * **관찰 후보**: 지지 접근·회복 대기 또는 시장·점수·손절폭으로 진입을 보류한 상태입니다. 관찰 표에 있다는 이유로 매수하지 않습니다.
    * **주문 취소**: 다음 거래일 미돌파, 지지 무효화 가격 침범, 최대 진입가 초과 시 취소하고 종가 후 다시 진단합니다. 지지 이탈과 돌파가 함께 있는 일봉은 시가에서 이미 체결된 경우 외에는 보수적으로 진입하지 않습니다.
    * **확정 일봉**: 당일 봉은 15:40 KST 이후 실제 조회된 데이터부터 사용합니다. 장중에 캐시된 봉은 시간이 지났다는 이유로 확정 봉으로 바꾸지 않습니다.

    #### ② 가격 4총사 (내비게이션)
    ```text
      [2차 목표가: +2.0R]  ──  최초 수량의 40% 매도 & 트레일링 스탑 가동
              ▲
      [1차 목표가: +1.0R]  ──  전체 수량의 30% 확정 매도 & 손절선을 매수가(본전)로 상향
              ▲
      [진입 기준가]        ──  내가 실제로 주식을 사야 하는 기준 가격
              ▼
      [동적 손절선]        ──  주가가 여기까지 떨어지면 미련 없이 전량 매도 (-1.0R)
    ```
    * **동적 손절선**: "내가 틀렸음을 인정하는 가격". 주가가 이 가격을 터치하면 미련 없이 전량 매도하여 큰 손실을 차단합니다.
    * **1차 목표가 (+1.0R)**: 손절폭만큼 수익이 난 지점. **가지고 있는 주식의 30%를 기계적으로 팔아 수익을 확정**하고, 남은 주식의 손절가를 **내가 산 가격(본전)**으로 수정합니다.
    * **2차 목표가 (+2.0R)**: 손절폭의 2배 수익이 난 지점. **최초 매수 수량의 40%를 추가 매도**하고, 나머지 30% 잔여분은 고점에서 1.5 ATR 밀릴 때까지 계속 들고 갑니다(트레일링 스탑).

    #### ③ 포지션 사이징: 왜 계좌 돈을 전부 안 쓰고 딱 'OO주'만 사라고 할까?
    * **1R 원칙**: "한 번의 매매에서 내 계좌 전체 돈의 딱 1%까지만 잃겠다"는 철칙입니다.
      * 예: 3,000만 원 계좌 $\rightarrow$ 거래비용 포함 계획 손실 예산은 **30만 원** (갭 손실 제외).
      * 손절폭이 3,000원이면? `30만 원 ÷ (3,000원 + 주당 비용)` 이내 매수.
      * 손절폭이 6,000원으로 크면? `30만 원 ÷ (6,000원 + 주당 비용)` 이내로 알아서 수량을 절반으로 줄여줍니다.
    * **수량 제한 요인**:
      * `RISK_BUDGET`: 설정한 손실 예산과 예상 거래비용에 맞춘 수량.
      * `WEIGHT_CAP`: 계좌 30% 비중 상한(몰빵 방지).
      * `LIQUIDITY_CAP`: 거래대금이 적은 종목에 갇히지 않도록 일 거래대금의 1% 이내로 제한.

    ---
    ### 3. STEP 3 & 4 백테스트 성적표 읽는 법
    "과거 2년 동안 이 규칙대로 매매했더라면 내 계좌가 어떻게 되었을까?"를 검증하는 성적표입니다.

    | 지표 이름 | 초보자 눈높이 설명 | 기준 합격선 |
    |---|---|---|
    | **승률 (Win Rate)** | 10번 매매해서 몇 번 수익을 냈는가? | **45% ~ 55%** 면 충분히 훌륭함 (벌 때 크게 벌기 때문) |
    | **일별 종가 MTM MDD** | 계좌 최고점 대비 가장 깊게 처박혔던 낙폭 | **-10% 이내**면 매우 우수, -15% 넘어가면 주의 |
    | **Profit Factor (손익비)** | 번 돈 총합 ÷ 잃은 돈 총합 | **1.5 이상**을 참고하되 표본 수·손실 거래·OOS 결과를 함께 확인 |
    | **기댓값 (Expectancy)** | 매매를 1번 완료할 때마다 계좌에 평균적으로 쌓이는 원화 | **플러스(+) 금액**이어야 함 |

    * **⚡ 3-모드 동시 삼각비교(Tricorder)**:
      * **보수적 (Worst-Case)**: 같은 날 목표가와 손절가를 둘 다 건드렸다면 **무조건 손절당했다고 가정한 결과**입니다.
      * **판단 팁**: 보수적 모드에서도 **수익률이 플러스이고 Profit Factor가 1.3 이상**을 유지한다면 추가 검증의 후보입니다. 작은 표본이나 생존 편향을 배제한 증거는 아닙니다.

    ---
    ### 4. 실전 매매 당일 행동 체크리스트
    1. **오전 8:30 (장 시작 전)**: 앱 상단 **시장 국면(Regime)**을 확인합니다. 빨간불(RISK-OFF)이면 매매하지 않고 쉽니다.
    2. **오전 8:40 (종목 진단)**: STEP 1에서 **진입 트랙**, **가격 4총사**, **권장 매수 주수**를 메모합니다.
    3. **오전 9:00 (장 시작)**: `REBOUND`의 트리거·최대 진입가·무효화 가격을 확인하고 다음 거래일 한 번만 유효한 조건부 주문을 검토합니다.
    4. **체결 직후 (필수 습관)**: 주식이 사지자마자 증권사 앱에서 **동적 손절선 가격에 '자동 감시 매도(Stop Loss)'**를 걸어둡니다.
    5. **장중 주가 상승 시**: 1차 목표가에 도달하면 30%를 팔고, 남은 주식의 손절가를 **매수가(본전)**로 수정합니다.
    """)

# -------------------------------------------------------------
# TAB 7: 시스템 무결성 헌장
# -------------------------------------------------------------
with tab7:
    st.subheader("📖 QUANT-EXECUTION 무결성 실행 원칙")
    st.markdown(r"""
    1. **단일/포트폴리오 엔진 완전 일치 (Unified Core Execution)**:
       * 단일 종목·포트폴리오 백테스트는 같은 지지·반등 조건, 진입 검사 및 포지션 청산 규칙을 사용합니다.
       * REBOUND 조건부 진입 체결부터 +1.5R 도달 시 +0.5R 보호 스탑 상향, +2.0R 도달 시 +1.0R 보호 스탑 상향, 1.5 ATR Trailing, Trend Break 청산까지 완전히 동일하게 작동합니다.
    2. **전체 자산 기준 리스크 사이징 (Portfolio Equity Risk Basis)**:
       * 포트폴리오 백테스트 시 슬롯 배분 현금이 아닌 현재 포트폴리오 전체 총평가자산($Cash + \sum Value_{open}$)을 기준으로 설정한 손실 예산과 단일 종목 30% 비중 상한을 계산하여 현실적 자본 할당을 구현합니다.
       * 포트폴리오 신규 매수에는 전체·동일 섹터·상관 종목군의 손절 위험 한도를 추가로 적용합니다. 갭 손실과 이후 평가액 변동은 한도를 초과할 수 있습니다.
    3. **전면 슬리피지 및 다단계 분할 회계 원칙 (Full Friction Accounting)**:
       * 최종 청산 가격으로 전체 수익률을 왜곡하지 않고, $T_1(30\%)$, $T_2(40\%)$, 잔여분의 실제 매도 유입 현금을 각각 집계하며, 지정가 익절(0.05%) 및 시장가 청산(0.10%) 슬리피지를 전면 차감하여 손익비를 계산합니다.
    4. **정합적 손절 통제 원칙 (Harmonized Risk Defense)**:
       * 가장 먼저 무너지는 핵심 지지선을 $max()$로 선택하되 최소 1.2 ATR 안전거리를 강제 확보하며, 손절폭이 -8.5% 이상인 종목은 거래 자체를 거부(WAIT)하여 자본 회전율을 방어합니다.
    5. **일별 종가 MTM 평가 기준 (Close-to-Close MTM Transparency)**:
       * 백테스트의 MDD 및 자산 곡선은 거래 종료 시점 기준이 아닌 매일 장 마감 종가 기준의 시가평가(MTM)로 측정되며, 장중 틱 단위 미세 낙폭은 배제된 일봉 기준 지표임을 명확히 규정합니다.
    6. **모호봉 삼각비교 원칙 (Tricorder Comparison Policy)**:
       * 당일 고가와 저가가 목표가와 손절가를 동시에 건드리는 모호한 일봉 발생 시, 보수적 모드(CONSERVATIVE), 중립 모드(NEUTRAL), 낙관 모드(OPTIMISTIC)를 동시 연산하여 안전 마진과 불확실성 델타를 명시합니다.
    7. **전 종목 3단 계층 상대강도 및 진성 3D WFO (Universal 3-Tier RS & Multi-D WFO)**:
       * KRX 전 종목 표준 업종을 5대 대표 ETF와 키워드 기반 추정 매핑하여 3단 상대강도를 매수 점수에 직접 반영하며, [Score x Trail x Hold] 27개 조합을 In-Sample에서 탐색한 뒤 60일 워밍업 정합성을 유지한 채 Out-of-Sample에 블라인드 동결 적용하여 전략의 영속성을 검증합니다.
    """)
