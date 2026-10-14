"""Streamlit interaction regressions with deterministic, offline market fixtures."""
import datetime
from pathlib import Path
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest


REGIME_DATE = '2026-10-14'
FIXTURES = '''
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
    if code == '005930':
        return pd.DataFrame()
    if code == '123456':
        raise RuntimeError('offline provider failure')
    dates = pd.bdate_range(end=market_today(), periods=80)
    close = np.linspace(10000, 11000, 80)
    return pd.DataFrame({'Open': close-20, 'High': close+100, 'Low': close-100,
                         'Close': close, 'Volume': 1000000}, index=dates)
load_stock_listing = fixture_listing
load_krx_listing = fixture_listing
load_price_history = fixture_prices
get_sector_for_stock = lambda *args, **kwargs: '대표지수'
load_sector_etf_series = lambda *args: pd.DataFrame()
build_sector_lookup = lambda *args: {}
def fixture_decision(*args, **kwargs):
    return {'action': 'BUY', 'badge': 'BUY',
            'scored': {'grade': 'BUY', 'score': 80, 'track': 'MARKET'},
            'levels': {'stop_price': 10000, 'loss_pct': -5,
                       'target_1r': 12000, 'target_2r': 13000}}
evaluate_full_execution_pipeline = fixture_decision
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
        today = now.astimezone(datetime.timezone(datetime.timedelta(hours=9))).date()
        self.app.session_state['screened_res'] = pd.DataFrame([
            {'종목명': 'SK하이닉스', '코드': '000660', '최종판정': 'BUY'}])
        self.app.session_state['screened_meta'] = {
            'context': (str(today), REGIME_DATE, 'FAVORABLE', 1.0, 2000, 100, False),
            'created_at': now.timestamp() - age,
            'counts': dict(evaluated=1, missing=0, failed=0, supply_missing=0),
            'errors': {},
        }
        self.app.run()
        self.assertFalse(self.app.exception)

    def test_transfer_updates_all_three_inputs_in_the_same_rerun(self):
        self.seed_results()
        self.button('선택 종목을 STEP 1으로 전송').click().run()
        self.assertFalse(self.app.exception)
        for key in ('diagnosis_stock', 'backtest_stock', 'stress_stock'):
            self.assertEqual(self.app.text_input(key=key).value, 'SK하이닉스')
        self.app.run()
        self.assertEqual(self.app.text_input(key='stress_stock').value, 'SK하이닉스')

    def test_expired_result_cannot_be_transferred_from_an_open_page(self):
        self.seed_results()
        metadata = dict(self.app.session_state['screened_meta'])
        metadata['created_at'] -= 3601
        self.app.session_state['screened_meta'] = metadata
        self.button('선택 종목을 STEP 1으로 전송').click().run()
        self.assertFalse(self.app.exception)
        self.assertNotIn('screened_res', self.app.session_state)
        self.assertEqual(self.app.text_input(key='diagnosis_stock').value, '삼성전자')

    def test_refresh_removes_previous_buy_results(self):
        self.seed_results()
        self.assertEqual(len(self.app.dataframe), 1)
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
        self.button('Buy Zone 종목 스캔 시작').click().run()
        self.assertFalse(self.app.exception)
        counts = self.app.session_state['screened_meta']['counts']
        self.assertEqual(counts, dict(evaluated=1, missing=1, failed=1, supply_missing=0))
        self.assertEqual(self.app.session_state['screened_meta']['errors'], {'RuntimeError': 1})
        self.assertEqual(len(self.app.session_state['screened_res']), 1)
        self.assertIn('시세기준일', self.app.session_state['screened_res'])
        self.app.run()
        self.assertEqual(len(self.app.dataframe), 1)


if __name__ == '__main__':
    unittest.main()
