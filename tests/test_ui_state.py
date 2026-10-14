"""Streamlit interaction regressions with deterministic, offline market fixtures."""
import datetime
from pathlib import Path
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest
from test_engine import load_engine


REGIME_DATE = '2026-10-14'
FIXTURES = '''
market_now = lambda: datetime.datetime.fromisoformat('2026-10-14T16:00:00+09:00')
market_today = lambda: market_now().date()
dual_regime = {
    'code': st.session_state.get('_test_regime', 'FAVORABLE'),
    'status': 'TEST', 'multiplier': 1.0, 'msg': 'offline fixture',
    'data_date': '2026-10-14',
}
def fixture_listing(*args):
    return pd.DataFrame({'Code': ['000660', '005930', '123456'],
                         'Name': ['SK하이닉스', '삼성전자', '실패종목'],
                         'Close': [10000]*3, 'Volume': [1000000]*3,
                         'Marcap': [1e12]*3, 'Amount': [1e10]*3})
def fixture_prices(code, *args):
    if st.session_state.get('_test_quote_missing', False):
        return pd.DataFrame()
    if code == '005930':
        return pd.DataFrame()
    if code == '123456':
        raise RuntimeError('offline provider failure')
    dates = pd.bdate_range(end=market_today(), periods=80)
    if st.session_state.get('_test_no_touch', False):
        close = np.linspace(10000, 11500, 80)
        return pd.DataFrame({'Open':close-20, 'High':close+100, 'Low':close-100,
                             'Close':close, 'Volume':1000000}, index=dates)
    idx = np.arange(80)
    close = 10000 + idx * 10
    return pd.DataFrame({'Open': close-30, 'High': close+np.where(idx % 3 == 0, 70, 10),
                         'Low': close-120, 'Close': close, 'Volume': 1000000}, index=dates)
load_stock_listing = fixture_listing
load_krx_listing = fixture_listing
load_price_history = fixture_prices
get_sector_for_stock = lambda *args, **kwargs: '대표지수'
load_sector_etf_series = lambda *args: pd.DataFrame()
build_sector_lookup = lambda *args: {}
find_stock_code = lambda *args: ('000660', 'SK하이닉스')
get_supply_data = lambda *args: {'status': 'FAIL', 'frgn_5': None}
'''


class UIStateTests(unittest.TestCase):
    def setUp(self):
        source = (Path(__file__).resolve().parents[1] / 'app.py').read_text()
        source = source.replace('dual_regime = get_current_dual_regime()', FIXTURES)
        self.app = AppTest.from_string(source, default_timeout=20).run()
        self.assertFalse(self.app.exception)

    def button(self, label):
        return next(b for b in self.app.button if b.label == label)

    def seed_results(self, age=0):
        now = datetime.datetime.now(datetime.timezone.utc)
        self.app.session_state['screened_res'] = pd.DataFrame([
            {'종목명': 'SK하이닉스', '코드': '000660', '최종판정': 'BUY'}])
        self.app.session_state['screened_meta'] = {
            'context': (REGIME_DATE, REGIME_DATE, 'FAVORABLE', 1.0, 2000, 100, False, 'SUPPORT-REBOUND-1'),
            'created_at': now.timestamp() - age,
            'counts': dict(evaluated=1, missing=0, failed=0, supply_missing=0),
            'errors': {},
        }
        self.app.run()
        self.assertFalse(self.app.exception)

    def test_transfer_updates_all_three_inputs_in_the_same_rerun(self):
        self.seed_results()
        self.button('이 종목의 매매 계획 보기').click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.text_input(key='diagnosis_stock').value, 'SK하이닉스')
        self.assertTrue(any('진입 대기' in item.value for item in self.app.success))
        next(c for c in self.app.checkbox if c.label == '고급 분석 보기').check().run()
        self.assertEqual(self.app.text_input(key='backtest_stock').value, 'SK하이닉스')
        next(s for s in self.app.selectbox if s.label == '분석 도구').set_value('WFO·슬리피지 검증').run()
        self.assertEqual(self.app.text_input(key='stress_stock').value, 'SK하이닉스')

    def test_expired_result_cannot_be_transferred_from_an_open_page(self):
        self.seed_results()
        metadata = dict(self.app.session_state['screened_meta'])
        metadata['created_at'] -= 3601
        self.app.session_state['screened_meta'] = metadata
        self.button('이 종목의 매매 계획 보기').click().run()
        self.assertFalse(self.app.exception)
        self.assertNotIn('screened_res', self.app.session_state)
        self.assertEqual(self.app.text_input(key='diagnosis_stock').value, '삼성전자')

    def test_refresh_removes_previous_buy_results(self):
        self.seed_results()
        self.assertGreaterEqual(len(self.app.dataframe), 1)
        self.button('시세·시장 데이터 새로 조회').click().run()
        self.assertFalse(self.app.exception)
        self.assertNotIn('screened_res', self.app.session_state)
        self.assertNotIn('screened_meta', self.app.session_state)
        self.assertEqual(len(self.app.dataframe), 0)

    def test_filter_market_change_and_expiration_invalidate_results(self):
        self.seed_results()
        cap = next(n for n in self.app.number_input if n.label == '시가총액 하한 (억원)')
        cap.set_value(2500).run()
        self.assertNotIn('screened_res', self.app.session_state)
        next(n for n in self.app.number_input if n.label == '시가총액 하한 (억원)').set_value(2000).run()
        self.seed_results(age=3601)
        self.assertNotIn('screened_res', self.app.session_state)
        self.seed_results()
        self.app.session_state['_test_regime'] = 'UNKNOWN'
        self.app.run()
        self.assertNotIn('screened_res', self.app.session_state)
        self.assertFalse(self.app.exception)

    def test_scan_preserves_counts_and_distinguishes_provider_failures(self):
        self.button('매수 후보 찾기').click().run()
        self.assertFalse(self.app.exception)
        counts = self.app.session_state['screened_meta']['counts']
        self.assertEqual(counts, dict(evaluated=1, missing=1, failed=1, supply_missing=0))
        self.assertEqual(self.app.session_state['screened_meta']['errors'], {'RuntimeError': 1})
        self.assertEqual(len(self.app.session_state['screened_res']), 1)
        self.assertIn('시세기준일', self.app.session_state['screened_res'])
        self.app.run()
        self.assertGreaterEqual(len(self.app.dataframe), 1)


    def test_risk_off_keeps_real_confirmed_pattern_as_observation(self):
        self.app.session_state['_test_regime'] = 'RISK-OFF'
        self.app.run()
        self.button('매수 후보 찾기').click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.session_state['screened_res'].empty)
        watch = self.app.session_state['screened_watch']
        self.assertEqual(len(watch), 1)
        self.assertEqual(watch['패턴상태'].iloc[0], '반등 확인 · 돌파 대기')
        self.assertIn('RISK-OFF', watch['보류·진입 근거'].iloc[0])
        self.assertEqual(self.app.session_state['screened_meta']['gates']['MARKET'], 1)
        self.button('이 종목의 매매 계획 보기').click().run()
        self.assertEqual(self.app.text_input(key='diagnosis_stock').value, 'SK하이닉스')
        self.button('매매 계획 확인').click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any('RISK-OFF' in item.value for item in self.app.error))
        self.button('시세·시장 데이터 새로 조회').click().run()
        self.assertNotIn('screened_watch', self.app.session_state)
        self.assertNotIn('screened_res', self.app.session_state)

    def test_unknown_market_preserves_observation_without_a_buy_plan(self):
        self.app.session_state['_test_regime'] = 'UNKNOWN'
        self.app.run()
        self.button('매수 후보 찾기').click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.session_state['screened_res'].empty)
        watch = self.app.session_state['screened_watch']
        self.assertEqual(len(watch), 1)
        self.assertIsNone(watch['진입트리거'].iloc[0])
        self.assertIsNone(watch['스코어'].iloc[0])
        self.assertIn('시장 상태', watch['보류·진입 근거'].iloc[0])

    def test_no_support_touch_is_an_observation_despite_a_high_score(self):
        self.app.session_state['_test_no_touch'] = True
        self.app.run()
        self.button('매수 후보 찾기').click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.session_state['screened_res'].empty)
        watch = self.app.session_state['screened_watch']
        self.assertEqual(len(watch), 1)
        self.assertEqual(watch['패턴상태'].iloc[0], '지지 접근')
        self.assertGreaterEqual(watch['스코어'].iloc[0], 70)

    def test_real_diagnosis_displays_next_session_trigger_and_sized_plan(self):
        self.button('매매 계획 확인').click().run()
        self.assertFalse(self.app.exception)
        labels = [metric.label for metric in self.app.metric]
        self.assertIn('조건부 진입 트리거', labels)
        self.assertTrue(any('진입 대기' in item.value for item in self.app.success))
        shares = next(m for m in self.app.metric if m.label == '최종 권장 매수 수량')
        self.assertNotEqual(shares.value, '0주')


    def test_displayed_shares_respect_budget_even_at_the_maximum_entry_price(self):
        self.button('매수 후보 찾기').click().run()
        row = self.app.session_state['screened_res'].iloc[0]
        next(s for s in self.app.slider if s.label == '1회 허용 손실률 (%)').set_value(.3).run()
        self.button('매매 계획 확인').click().run()
        self.assertFalse(self.app.exception)
        value = next(m.value for m in self.app.metric if m.label == '최종 권장 매수 수량')
        shares = int(value.replace(',', '').replace('주', ''))
        risk = load_engine()['expected_stop_risk'](row['최대진입가'], row['손절가'])
        self.assertGreater(shares, 0)
        self.assertLessEqual(shares * risk, 30000000 * .003)


if __name__ == '__main__':
    unittest.main()
