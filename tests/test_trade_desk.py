"""Two-view user journey with actual-fill recording and no broker side effects."""
import datetime
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest
from test_engine import load_engine
from test_ui_state import FIXTURES


class TradeDeskTests(unittest.TestCase):
    def setUp(self):
        source = (Path(__file__).resolve().parents[1] / 'app.py').read_text()
        self.app = AppTest.from_string(source.replace('dual_regime = get_current_dual_regime()', FIXTURES), default_timeout=20).run()
        self.assertFalse(self.app.exception)

    def button(self, label):
        return next(b for b in self.app.button if b.label == label)

    def holding(self):
        engine = load_engine()
        engine['market_today'] = lambda: datetime.date(2026,10,14)
        return engine['create_holding']('000660','SK하이닉스',10000,100,'2026-10-12',9500)

    def show_holdings(self):
        self.app.radio(key='main_view').set_value('내 보유종목 관리').run()
        self.assertFalse(self.app.exception)

    def test_default_has_two_views_and_advanced_analyses_are_lazy(self):
        self.assertEqual(self.app.radio(key='main_view').options,['오늘의 매수 후보','내 보유종목 관리'])
        labels = [b.label for b in self.app.button]
        self.assertNotIn('일별 MTM 백테스트 가동',labels)
        self.assertFalse(any('WFO' in label for label in labels))
        next(c for c in self.app.checkbox if c.label == '고급 분석 보기').check().run()
        self.assertIn('일별 MTM 백테스트 가동',[b.label for b in self.app.button])
        self.assertFalse(self.app.exception)

    def test_candidate_to_plan_to_holding_requires_actual_fill_confirmation(self):
        self.button('매수 후보 찾기').click().run()
        self.button('이 종목의 매매 계획 보기').click().run()
        self.assertEqual(self.app.session_state['holding_book'],[])
        self.button('이 종목의 매수 체결 입력').click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.radio(key='main_view').value,'내 보유종목 관리')
        self.assertEqual(self.app.session_state['holding_book'],[])
        self.assertEqual(self.app.text_input(key='holding_symbol').value,'000660')
        self.app.number_input(key='holding_entry_price').set_value(10000.)
        self.app.number_input(key='holding_quantity').set_value(100)
        self.app.number_input(key='holding_initial_stop').set_value(9500.)
        self.app.date_input(key='holding_entry_date').set_value(datetime.date(2026,10,12))
        self.button('매수 체결 저장').click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.session_state['holding_book']),1)
        self.assertEqual(self.app.session_state['holding_book'][0]['quantity'],100)
        self.assertTrue(any('1차 부분 익절' in w.value for w in self.app.warning))

    def test_actual_sale_changes_balance_and_navigation_does_not_lose_it(self):
        self.app.session_state['holding_book'] = [self.holding()]
        self.show_holdings()
        self.assertEqual(self.app.session_state['holding_book'][0]['fills'],[])
        # The target was reached; rerendering must not auto-record a sale.
        self.app.run()
        self.assertEqual(self.app.session_state['holding_book'][0]['fills'],[])
        qty = next(n for n in self.app.number_input if n.label == '실제 매도 수량 (주)')
        self.assertEqual(qty.value,30)
        qty.set_value(30)
        next(n for n in self.app.number_input if n.label == '실제 매도가 (원)').set_value(10500.)
        next(s for s in self.app.selectbox if s.label == '매도 구분').set_value('1차 익절')
        self.button('매도 체결 저장').click().run()
        self.assertFalse(self.app.exception)
        book = self.app.session_state['holding_book']
        self.assertEqual(book[0]['fills'][0]['quantity'],30)
        self.assertEqual(load_engine()['holding_balance'](book[0]),70)
        self.assertEqual(book[0]['stop_updates'],[])
        self.app.radio(key='main_view').set_value('오늘의 매수 후보').run()
        self.show_holdings()
        self.assertEqual(load_engine()['holding_balance'](self.app.session_state['holding_book'][0]),70)
        self.button('마지막 매도 입력 취소').click().run()
        self.assertEqual(load_engine()['holding_balance'](self.app.session_state['holding_book'][0]),100)
        self.assertFalse(self.app.exception)

    def test_quote_failure_and_market_risk_do_not_remove_actual_positions(self):
        position = self.holding()
        self.app.session_state['holding_book'] = [position]
        self.app.session_state['_test_regime'] = 'RISK-OFF'
        self.app.session_state['_test_quote_missing'] = True
        self.show_holdings()
        self.assertEqual(self.app.session_state['holding_book'],[position])
        self.assertTrue(any('시세 확인 필요' in w.value for w in self.app.warning))
        self.button('보유 시세 새로 확인').click().run()
        self.assertEqual(self.app.session_state['holding_book'],[position])
        self.assertFalse(self.app.exception)

    def test_invalid_actual_fill_is_not_saved(self):
        self.show_holdings()
        self.app.text_input(key='holding_symbol').set_value('000660')
        self.app.number_input(key='holding_entry_price').set_value(10000.)
        self.app.number_input(key='holding_initial_stop').set_value(11000.)
        self.button('매수 체결 저장').click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state['holding_book'],[])
        self.assertTrue(any('최초 손절가' in error.value for error in self.app.error))

    def test_sessions_do_not_share_the_actual_holdings_book(self):
        self.app.session_state['holding_book'] = [self.holding()]
        self.show_holdings()
        source = (Path(__file__).resolve().parents[1] / 'app.py').read_text()
        other = AppTest.from_string(source.replace('dual_regime = get_current_dual_regime()', FIXTURES), default_timeout=20).run()
        self.assertEqual(other.session_state['holding_book'],[])
        self.assertEqual(len(self.app.session_state['holding_book']),1)


if __name__ == '__main__':
    unittest.main()
