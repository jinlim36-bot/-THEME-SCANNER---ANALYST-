import streamlit as st
import pandas as pd
import datetime
import requests
import io
import FinanceDataReader as fdr
from google import genai

# =============================================================
# [API 및 모델 설정]
# =============================================================
DEFAULT_API_KEY = "여기에_AQ로_시작하는_키를_붙여넣으세요"
GEMINI_MODEL = "gemini-3.6-flash"  # 최신 권장 모델 지정

# -------------------------------------------------------------
# 0. UI 설정 및 클라이언트 초기화
# -------------------------------------------------------------
st.set_page_config(
    page_title="종목 검색기 (THEME SCANNER - ANALYST)",
    page_icon="📈",
    layout="wide"
)

st.sidebar.header("🔑 Gemini API 설정")
input_key = st.sidebar.text_input(
    "API Key 입력",
    value="" if "여기에" in DEFAULT_API_KEY else DEFAULT_API_KEY,
    type="password",
    help="AQ.로 시작하는 59자리 키를 입력하세요."
)

active_key = input_key.strip() if input_key.strip() else (DEFAULT_API_KEY if "여기에" not in DEFAULT_API_KEY else "")

if not active_key:
    st.warning("👈 왼쪽 사이드바에 발급받으신 Gemini API 키(AQ....)를 붙여넣어 주세요.")
    st.stop()

# Gemini SDK 클라이언트 생성
try:
    client = genai.Client(api_key=active_key)
except Exception as e:
    st.error(f"API 클라이언트 초기화 오류: {e}")
    st.stop()

# -------------------------------------------------------------
# [보조 함수] 네이버 금융에서 외인/기관 20거래일 누적 수급 집계
# -------------------------------------------------------------
def get_investor_sentiment(code):
    url = f"https://finance.naver.com/item/frgn.naver?code={code}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        res = requests.get(url, headers=headers, timeout=5)
        dfs = pd.read_html(io.StringIO(res.text), encoding="cp949")
        df_target = None
        for table in dfs:
            if "기관" in str(table.columns) and "외국인" in str(table.columns):
                df_target = table
                break
        
        if df_target is not None:
            df_target.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_target.columns]
            df_target = df_target.dropna(subset=[df_target.columns[0]])
            
            inst_col = [c for c in df_target.columns if "기관" in c and "순매" in c][0]
            frgn_col = [c for c in df_target.columns if "외국인" in c and "순매" in c][0]
            
            inst_sum = pd.to_numeric(df_target[inst_col].astype(str).str.replace(',', ''), errors='coerce').head(20).sum()
            frgn_sum = pd.to_numeric(df_target[frgn_col].astype(str).str.replace(',', ''), errors='coerce').head(20).sum()
            return int(frgn_sum), int(inst_sum)
    except Exception:
        pass
    return 0, 0

st.title("📈 종목 검색기 (THEME SCANNER - ANALYST)")
st.caption(f"ENGINE: Google {GEMINI_MODEL} | DATA: Naver Finance, KRX Open Feed")

tab1, tab2, tab3 = st.tabs(["STEP 1 · 수급 테마 검색", "STEP 2 · 종목 분석", "📖 사용 매뉴얼"])

# -------------------------------------------------------------
# STEP 1: 수급 테마 검색
# -------------------------------------------------------------
with tab1:
    st.subheader("수급 테마 검색 (Screener)")
    st.caption("전체 시장에서 자금 유입(거래량 급증)과 상승 탄력이 발생한 종목을 고속 추출합니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        min_return = st.slider("5일 누적 하한 (%)", min_value=0, max_value=30, value=5)
        candidate_pool = st.slider("탐색 후보군 (거래대금 상위)", min_value=30, max_value=200, value=70)
    with col2:
        vol_multiple = st.slider("거래량 배수 (전일 대비)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
        min_market_cap = st.number_input("시총 하한 (억원)", value=1000, step=500)

    if st.button("수급 필터링 실행", type="primary"):
        with st.spinner("네이버 금융 & KRX 시장 데이터 일괄 수집 중..."):
            try:
                df_krx = fdr.StockListing('KRX')
                df_filtered = df_krx[df_krx['Marcap'] >= (min_market_cap * 100000000)].copy()
                df_filtered = df_filtered.sort_values(by="Amount", ascending=False).head(candidate_pool)
                
                screened_stocks = []
                start_check_date = (datetime.datetime.today() - datetime.timedelta(days=15)).strftime("%Y-%m-%d")
                
                progress_bar = st.progress(0)
                total_len = len(df_filtered)

                for idx, (_, row) in enumerate(df_filtered.iterrows()):
                    progress_bar.progress((idx + 1) / total_len)
                    code = row['Code']
                    name = row['Name']
                    
                    try:
                        hist = fdr.DataReader(code, start_check_date)
                        if len(hist) < 5:
                            continue
                        
                        last_vol = hist['Volume'].iloc[-1]
                        prev_vol = hist['Volume'].iloc[-2] if hist['Volume'].iloc[-2] > 0 else 1
                        vol_ratio = last_vol / prev_vol
                        return_5d = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5]) * 100
                        cap_val = int(row['Marcap'] / 100000000)

                        if return_5d >= min_return and vol_ratio >= vol_multiple:
                            screened_stocks.append({
                                "코드": code,
                                "종목명": name,
                                "현재가": int(hist['Close'].iloc[-1]),
                                "5일수익률(%)": round(return_5d, 2),
                                "거래량배수": round(vol_ratio, 2),
                                "시가총액(억)": cap_val
                            })
                    except Exception:
                        continue
                
                progress_bar.empty()
                res_df = pd.DataFrame(screened_stocks)
                st.session_state["screened_df"] = res_df
                st.success(f"필터링 완료! 발굴된 주도 후보군: {len(res_df)}개")

            except Exception as e:
                st.error(f"데이터 수집 중 오류: {e}")

    if "screened_df" in st.session_state and not st.session_state["screened_df"].empty:
        df_display = st.session_state["screened_df"]
        st.dataframe(df_display, use_container_width=True)

        stock_names = df_display["종목명"].tolist()
        theme_prompt = f"""
        당신은 국내 최정상급 헤지펀드 애널리스트입니다. 아래 거래량과 수급이 급증한 주도 후보 종목들을 분석하여 
        현재 시장을 주도하는 테마군으로 묶고 각 테마의 대장주와 수혜주를 명확히 분류해 주세요.

        [분석 대상 종목 목록]
        {', '.join(stock_names)}

        [출력 양식]
        각 테마별로 아래 포맷을 엄격히 지켜 작성할 것:
        - 테마명: [테마 이름]
        - 거시 트리거 / 이슈: (자금 유입의 핵심 재료 및 모멘텀)
        - 대장주: (테마 내 시세를 견인하는 핵심 1~2종목)
        - 수혜주: (동반 상승 탄력이 기대되는 종목)
        """
        
        if st.button("테마 및 주도주 AI 분류 (Gemini)"):
            with st.spinner("Gemini가 실시간 주도 테마와 대장주를 분류하고 있습니다..."):
                try:
                    response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=theme_prompt
                    )
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"분류 생성 중 오류 발생: {e}")

# -------------------------------------------------------------
# STEP 2: 종목 심층 분석
# -------------------------------------------------------------
with tab2:
    st.subheader("종목 심층 분석 (AI Analyst)")
    st.caption("기업 펀더멘털, 차트 위치, 외인/기관 20일 수급을 집계하여 AI 7단계 정밀 리포트를 제공합니다.")
    
    target_stock = st.text_input("분석할 종목명을 입력하세요 (예: 삼성전자, SK하이닉스)")

    if st.button("종목 분석하기", type="primary"):
        if not target_stock.strip():
            st.warning("종목명을 입력해주세요.")
        else:
            with st.spinner(f"'{target_stock}'의 시장 데이터 집계 및 Gemini 심층 분석 중..."):
                try:
                    df_krx = fdr.StockListing('KRX')
                    matched = df_krx[df_krx['Name'] == target_stock.strip()]
                    
                    if matched.empty:
                        st.error("존재하지 않는 종목명이거나 유효하지 않은 이름입니다.")
                    else:
                        target_code = matched.iloc[0]['Code']
                        start_hist = (datetime.datetime.today() - datetime.timedelta(days=45)).strftime("%Y-%m-%d")
                        
                        df_price = fdr.DataReader(target_code, start_hist)
                        curr_price = int(df_price['Close'].iloc[-1])
                        high_20d = int(df_price['High'].tail(20).max())
                        low_20d = int(df_price['Low'].tail(20).min())

                        foreign_buy_qty, inst_buy_qty = get_investor_sentiment(target_code)

                        analysis_prompt = f"""
                        당신은 수급과 모멘텀을 전문으로 다루는 헤지펀드 트레이더이자 리서치 센터장입니다. 
                        제공된 정량 데이터를 바탕으로 아래 7단계 규격에 맞춰 엄밀하고 직관적인 종목 리포트를 작성하세요.

                        [입력 데이터]
                        - 종목명: {target_stock} (종목코드: {target_code})
                        - 현재 종가: {curr_price:,}원
                        - 최근 20거래일 최고가: {high_20d:,}원 / 최저가: {low_20d:,}원
                        - 최근 20거래일 외국인 누적 순매수량: {foreign_buy_qty:,}주
                        - 최근 20거래일 기관 누적 순매수량: {inst_buy_qty:,}주

                        [출력 형식 - 각 번호와 항목명을 그대로 유지할 것]
                        1단계. 종목 기본 정보 (비즈니스 모델, 주요 매출원)
                        2단계. 최근 시장 재료 및 공시 분석 (섹터 동향 및 호재/악재)
                        4단계. 외국인·기관 수급 평가 (순매수 수량 기반 메이저 자금의 매집 강도 진단)
                        5단계. 차트와 가격 위치 (20일 변동폭 내 위치 및 핵심 지지/저항 라인)
                        6단계. 리스크 요인 (변동성, 업황 불확실성, 밸류에이션 등)
                        7단계. 최종 판단:
                          - [판단 결과]: (적극매수 / 분할매수 / 관망 / 비중축소 중 1개 선택)
                          - [판단 근거]: (수급 및 모멘텀 기반 핵심 요약)
                          - [매매 시나리오]:
                              * 매수 적정가: (구체적 가격대)
                              * 목표가: (단기/스윙 목표 가격)
                              * 손절 기준선: (이탈 시 대응할 손절 가격)
                        """

                        response = client.models.generate_content(
                            model=GEMINI_MODEL,
                            contents=analysis_prompt
                        )
                        st.markdown(response.text)

                except Exception as e:
                    st.error(f"분석 중 오류 발생: {e}")

# -------------------------------------------------------------
# STEP 3: 사용 매뉴얼
# -------------------------------------------------------------
with tab3:
    st.subheader("📖 종목 검색기 공식 사용 매뉴얼")
    st.markdown("""
    이 시스템은 감(Feeling)에 의존하는 매매를 배제하고, **자금 유입(거래량·거래대금)과 외국인·기관의 수급**을 기반으로 주도 테마와 핵심 종목을 발굴하도록 설계되었습니다.
    """)
    st.divider()

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("### 🎯 STEP 1. 수급 테마 검색 활용법")
        st.markdown("""
        * **5일 누적 하한 (%)**: `5% ~ 10%` 추천 (상승 탄력이 붙기 시작한 초입 포착)
        * **거래량 배수**: `2.0배 이상` 권장 (평소 대비 유의미한 거래대금 폭증 종목)
        * **시총 하한**: 최소 `1,000억 원` 이상 (유동성이 적은 품절주/작전주 필터링)
        * **테마 및 주도주 AI 분류**: 추출된 종목군을 클러스터링하여 시장 주도 테마 및 1등 대장주 자동 판별
        """)
    with col_m2:
        st.markdown("### 🔍 STEP 2. 종목 심층 분석 활용법")
        st.markdown("""
        * **수급 유입 진단**: 4단계에서 최근 20일간 외인·기관의 순매수 기조 확인
        * **가격 밴드 위치**: 5단계에서 20일 고점/저점 대비 현 주가의 가격 매력도 점검
        * **시나리오 매매**: 7단계의 [분할 매수 적정가 / 목표가 / 손절 기준선]을 기준으로 기계적 매매 대응
        """)