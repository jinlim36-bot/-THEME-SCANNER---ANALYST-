"""Pattern gates, next-session orders and live snapshot integrity regressions."""
import datetime
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from test_engine import load_engine, rebound_history


def no_touch_history():
    dates = pd.bdate_range('2024-01-01', periods=80)
    close = np.linspace(10000, 11500, 80)
    return pd.DataFrame({'Open': close-20, 'High': close+100, 'Low': close-100,
                         'Close': close, 'Volume': 1000000}, index=dates)


class SupportReboundTests(unittest.TestCase):
    def setUp(self):
        self.e = load_engine()
        self.stock = rebound_history(80)
        self.market = pd.DataFrame({'Close': 3000.}, index=self.stock.index)

    def decision(self, data=None, regime='FAVORABLE', **kwargs):
        return self.e['evaluate_full_execution_pipeline'](
            self.stock if data is None else data, regime,
            kospi_slice=self.market, sector_slice=self.market, **kwargs)

    def execute(self, setup, opening=None, high=None, low=None, date=None, slippage=.001):
        return self.e['check_3track_execution'](
            'REBOUND', setup['trigger_price']-20 if opening is None else opening,
            setup['trigger_price']+30 if high is None else high,
            setup['invalidation_price']+30 if low is None else low,
            0, 0, slippage_rate=slippage, entry_conditions=setup,
            execution_date=pd.Timestamp(setup['signal_date'])+pd.offsets.BDay(1) if date is None else date)

    def test_confirmed_pattern_has_frozen_reference_and_next_day_trigger(self):
        result = self.decision()
        setup = result['pattern']
        self.assertEqual(result['action'], 'BUY')
        self.assertEqual(result['scored']['track'], 'REBOUND')
        self.assertEqual(setup['state'], 'CONFIRMED')
        self.assertEqual(setup['reference_date'], str(self.stock.index[-2].date()))
        self.assertGreater(setup['trigger_price'], self.stock.High.iloc[-1])
        self.assertEqual(result['levels']['entry_price'], setup['trigger_price'])
        self.assertLessEqual(result['levels']['stop_price'], setup['invalidation_price'])
        self.assertTrue(self.execute(setup)[0])

    def test_score_cannot_replace_a_missing_support_test(self):
        result = self.decision(no_touch_history())
        self.assertGreaterEqual(result['scored']['score'], 70)
        self.assertEqual(result['pattern']['state'], 'APPROACH')
        self.assertEqual(result['action'], 'WAIT')
        self.assertEqual(result['block_code'], 'PATTERN')

    def test_previous_false_buy_below_ma20_is_now_blocked(self):
        data = no_touch_history()
        price = data.Close.iloc[-21:-1].mean() * .995
        data.iloc[-1] = [price-30, price+10, price-50, price, 1500000]
        result = self.decision(data)
        self.assertGreaterEqual(result['scored']['score'], 70)
        self.assertLess(data.Close.iloc[-1], data.Close.iloc[-2])
        self.assertLess(result['scored']['disparity_pct'], 0)
        self.assertNotIn(result['action'], {'BUY', 'BUY_REDUCED'})

    def test_recovered_support_without_previous_close_recovery_waits(self):
        data = self.stock.copy()
        price = data.Close.iloc[-2] - 10
        data.iloc[-1] = [price-30, price+10, price-120, price, 1000000]
        result = self.decision(data)
        self.assertEqual(result['pattern']['state'], 'RECOVERED')
        self.assertEqual(result['action'], 'WAIT')

    def test_support_break_and_zero_volume_cannot_confirm(self):
        setup = self.decision()['pattern']
        data = self.stock.copy()
        price = setup['invalidation_price']-10
        data.iloc[-1] = [price-30, price+10, price-120, price, 1500000]
        self.assertEqual(self.decision(data)['pattern']['state'], 'BROKEN')
        data = self.stock.copy()
        data.loc[data.index[-1], 'Volume'] = 0
        self.assertEqual(self.decision(data)['pattern']['state'], 'RECOVERED')
        self.assertEqual(self.decision(data)['action'], 'WAIT')

    def test_market_permission_never_erases_confirmed_observation(self):
        for regime, action, multiplier in [('FAVORABLE','BUY',1.), ('NEUTRAL','BUY_REDUCED',.5),
                                            ('RISK-OFF','AVOID',0.), ('UNKNOWN','WAIT',0.)]:
            with self.subTest(regime=regime):
                result = self.decision(regime=regime)
                self.assertEqual(result['pattern']['state'], 'CONFIRMED')
                self.assertEqual(result['action'], action)
                self.assertEqual(result['multiplier'], multiplier)

    def test_score_and_stop_width_gates_still_apply(self):
        result = self.decision(min_score=90)
        self.assertEqual(result['block_code'], 'SCORE')
        data = self.stock.copy()
        data.loc[data.index[:-1], 'High'] = data.Close.iloc[:-1] + 1000
        data.loc[data.index[:-1], 'Low'] = data.Close.iloc[:-1] - 1000
        result = self.decision(data)
        self.assertEqual(result['pattern']['state'], 'CONFIRMED')
        self.assertEqual(result['block_code'], 'STOP')
        self.assertEqual(result['multiplier'], 0)

    def test_current_signal_day_and_stale_order_cannot_fill(self):
        setup = self.decision()['pattern']
        for date in [pd.Timestamp(setup['signal_date']), pd.Timestamp(setup['signal_date'])+pd.Timedelta(days=8)]:
            self.assertFalse(self.execute(setup, date=date)[0])
        self.assertFalse(self.execute(setup, high=setup['trigger_price']-10)[0])

    def test_order_is_valid_only_on_the_symbols_next_session(self):
        decision = self.decision()
        first = self.stock.index[-1] + pd.offsets.BDay(1)
        second = first + pd.offsets.BDay(1)
        dates = self.stock.index.append(pd.DatetimeIndex([first, second]))
        setup = self.e['entry_order_conditions'](decision, dates)
        self.assertTrue(self.execute(setup, date=first)[0])
        self.assertFalse(self.execute(setup, date=second)[0])
        self.assertNotIn('valid_for_date', decision['pattern'])

    def test_gap_break_gap_chase_and_slippage_over_ceiling_cancel(self):
        setup = self.decision()['pattern']
        cases = [dict(opening=setup['invalidation_price']-10, low=setup['invalidation_price']-20),
                 dict(opening=setup['max_entry_price']+10, high=setup['max_entry_price']+20),
                 dict(opening=setup['max_entry_price'], high=setup['max_entry_price']+20)]
        for case in cases:
            with self.subTest(case=case):
                self.assertFalse(self.execute(setup, **case)[0])

    def test_intraday_break_and_trigger_ambiguity_cancel_but_known_open_fill_counts(self):
        setup = self.decision()['pattern']
        self.assertFalse(self.execute(setup, low=setup['invalidation_price']-10)[0])
        self.assertTrue(self.execute(setup, opening=setup['trigger_price'], low=setup['invalidation_price']-10)[0])
        data = self.stock.copy()
        next_day = data.index[-1] + pd.offsets.BDay(1)
        data.loc[next_day] = [setup['trigger_price'], setup['trigger_price']+50,
                              setup['invalidation_price']-100, setup['trigger_price']-50, 1000000]
        regime = pd.DataFrame({'Regime': 'FAVORABLE', 'KS_Close': 3000.}, index=data.index)
        _, trades, _ = self.e['run_daily_mtm_backtest'](data, regime, eval_start_date=next_day)
        self.assertEqual(len(trades), 1)
        self.assertLess(trades['실현손익(원)'].iloc[0], 0)

    def test_unfilled_next_session_order_does_not_carry_forward(self):
        setup = self.decision()['pattern']
        data = self.stock.copy()
        first = data.index[-1] + pd.offsets.BDay(1)
        second = first + pd.offsets.BDay(1)
        data.loc[first] = [setup['trigger_price']-30, setup['trigger_price']-10,
                           setup['trigger_price']-80, setup['trigger_price']-40, 1000000]
        data.loc[second] = [setup['trigger_price']+10, setup['trigger_price']+30,
                            setup['trigger_price']-20, setup['trigger_price']+20, 1000000]
        regime = pd.DataFrame({'Regime': 'FAVORABLE', 'KS_Close': 3000.}, index=data.index)
        original = self.e['evaluate_full_execution_pipeline']
        def only_first_signal(frame, *args, **kwargs):
            return original(frame, *args, **kwargs) if frame.index[-1] == self.stock.index[-1] else self.e['blocked_decision']('No new signal')
        self.e['evaluate_full_execution_pipeline'] = only_first_signal
        daily, trades, _ = self.e['run_daily_mtm_backtest'](data, regime, eval_start_date=first)
        self.assertTrue(trades.empty)
        self.assertEqual(daily.Stock_Value.max(), 0)

    def test_future_bars_do_not_change_past_patterns_or_equity(self):
        data = rebound_history(150)
        changed = data.copy()
        changed.loc[changed.index[110:], ['Open','High','Low','Close']] *= 2
        prepared = self.e['prepare_engine_history'](changed)
        for end in [80, 100, 110]:
            self.assertEqual(self.e['detect_support_rebound'](data.iloc[:end]),
                             self.e['detect_support_rebound'](prepared.iloc[:end]))
        regime = pd.DataFrame({'Regime':'FAVORABLE','KS_Close':3000.}, index=data.index)
        left = self.e['run_daily_mtm_backtest'](data, regime)[0]
        right = self.e['run_daily_mtm_backtest'](changed, regime)[0]
        pd.testing.assert_frame_equal(left.loc[:data.index[109]], right.loc[:data.index[109]], check_exact=True)

    def test_live_stale_stock_is_blocked_even_with_matching_historical_benchmark(self):
        now = datetime.datetime(2024,4,19,16,tzinfo=datetime.timezone(datetime.timedelta(hours=9)))
        self.e['market_now'] = lambda: now
        self.e['market_today'] = lambda: now.date()
        regime = {'code':'FAVORABLE','data_date':'2024-04-19'}
        result, _ = self.e['evaluate_live_execution'](self.stock.iloc[:-5], regime, self.market, self.market)
        self.assertEqual(result['block_code'], 'DATA')
        self.assertIn('마지막 거래일', result['reason'])

    def test_cached_intraday_snapshot_never_becomes_a_completed_bar_by_waiting(self):
        zone = datetime.timezone(datetime.timedelta(hours=9))
        now = datetime.datetime(2024,4,19,16,tzinfo=zone)
        data = self.stock.copy()
        data.attrs['retrieved_at'] = datetime.datetime(2024,4,19,15,20,tzinfo=zone).isoformat()
        result = self.e['completed_live_history'](data, now)
        self.assertEqual(result.index[-1], data.index[-2])
        data.attrs['retrieved_at'] = datetime.datetime(2024,4,19,15,40,tzinfo=zone).isoformat()
        self.assertEqual(self.e['completed_live_history'](data, now).index[-1], data.index[-1])
        self.assertEqual(self.e['completed_live_history'](self.stock, now.replace(hour=14)).index[-1], data.index[-2])

    def test_live_market_regime_dates_must_match(self):
        self.e['market_today'] = lambda: self.stock.index[-1].date()
        issue = self.e['live_snapshot_issue'](self.stock, self.market, '2024-04-18')
        self.assertIn('기준일이 다릅니다', issue)

    def test_live_supply_uses_only_the_completed_signal_date(self):
        self.e['io'] = io
        self.e['requests'] = SimpleNamespace(get=lambda *a, **k: SimpleNamespace(status_code=200, content=b'fixture'))
        table = pd.DataFrame({'날짜': pd.date_range('2024-04-14', periods=6)[::-1],
                              '외국인_순매매량': [9999,10,20,30,40,50]})
        with patch.object(pd, 'read_html', return_value=[table]):
            result = self.e['get_supply_data']('005930', '2024-04-18')
        self.assertEqual(result, {'status':'OK','frgn_5':150})
        with patch.object(pd, 'read_html', return_value=[table.drop(columns='날짜')]):
            self.assertEqual(self.e['get_supply_data']('005930', '2024-04-18')['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
