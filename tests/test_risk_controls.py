"""Data failures, concentration limits and statistical confidence regressions."""
import datetime
import unittest

import numpy as np
import pandas as pd
from test_engine import history, load_engine


class RiskControlTests(unittest.TestCase):
    def setUp(self):
        self.e = load_engine()
        self.df = history()
        self.reg = pd.DataFrame({'Regime': 'FAVORABLE', 'KS_Close': self.df.Close}, index=self.df.index)

    def test_missing_live_market_never_becomes_neutral(self):
        self.e['load_historical_dual_regime'] = lambda *args: pd.DataFrame()
        result = self.e['get_current_dual_regime']()
        self.assertEqual(result['code'], 'UNKNOWN')
        self.assertEqual(result['multiplier'], 0)
        decision = self.e['evaluate_full_execution_pipeline'](
            self.df, result['code'], kospi_slice=self.df[['Close']])
        self.assertEqual(decision['multiplier'], 0)
        self.assertEqual(decision['scored']['grade'], 'UNVERIFIED')

    def test_stale_and_future_live_indices_are_blocked(self):
        today = datetime.date(2026, 10, 13)
        self.e['market_today'] = lambda: today
        for delta in [-8, 1]:
            data = pd.DataFrame({'Regime': ['FAVORABLE'], 'KS_Close': [3000]},
                                index=[pd.Timestamp(today) + pd.Timedelta(days=delta)])
            self.e['load_historical_dual_regime'] = lambda *args: data
            self.assertEqual(self.e['get_current_dual_regime']()['code'], 'UNKNOWN')

    def test_dual_regime_requires_both_indices_and_warmup(self):
        self.assertTrue(self.e['build_dual_regime'](self.df, pd.DataFrame()).empty)
        trend = self.df.copy()
        trend['Close'] = 3000 + np.arange(len(trend)) * 5
        result = self.e['build_dual_regime'](trend, trend)
        self.assertTrue((result.Regime.iloc[:24] == 'UNKNOWN').all())
        self.assertEqual(result.Regime.iloc[-1], 'FAVORABLE')
        bad = trend.copy()
        bad.loc[bad.index[-1], 'Close'] = np.nan
        self.assertEqual(self.e['build_dual_regime'](trend, bad).Regime.iloc[-1], 'UNKNOWN')

    def test_historic_missing_regime_does_not_inherit_prior_state(self):
        date = self.df.index[100]
        reg = self.reg.drop(date)
        self.assertEqual(self.e['regime_at'](reg, date), 'UNKNOWN')
        self.assertEqual(self.e['regime_at'](reg, self.df.index[99]), 'FAVORABLE')

    def test_invalid_ohlcv_or_missing_benchmark_blocks_entry(self):
        for benchmark in [None, self.df[['Close']].iloc[:-1]]:
            result = self.e['evaluate_full_execution_pipeline'](
                self.df, 'FAVORABLE', kospi_slice=benchmark)
            self.assertEqual(result['multiplier'], 0)
        for column, value in [('Close', np.nan), ('Volume', -1), ('Low', 1e9)]:
            bad = self.df.copy()
            bad.loc[bad.index[-1], column] = value
            result = self.e['evaluate_full_execution_pipeline'](
                bad, 'FAVORABLE', kospi_slice=self.df[['Close']])
            self.assertEqual(result['scored']['grade'], 'UNVERIFIED')

    def test_ma60_cannot_be_replaced_with_ma20_for_new_entries(self):
        short = self.df.iloc[:59]
        result = self.e['evaluate_full_execution_pipeline'](
            short, 'FAVORABLE', kospi_slice=short[['Close']])
        self.assertEqual(result['multiplier'], 0)
        self.assertIn('60거래일', result['reason'])

    def test_stock_must_outperform_its_sector_for_full_rs_points(self):
        dates = self.df.index[-21:]
        stock = pd.DataFrame({'Close': np.linspace(100, 110, 21)}, index=dates)
        market = pd.DataFrame({'Close': np.full(21, 100)}, index=dates)
        stronger_sector = pd.DataFrame({'Close': np.linspace(100, 120, 21)}, index=dates)
        weaker_sector = pd.DataFrame({'Close': np.linspace(100, 103, 21)}, index=dates)
        self.assertEqual(self.e['calculate_3tier_rs'](stock, market, stronger_sector)[0], 5)
        self.assertEqual(self.e['calculate_3tier_rs'](stock, market, weaker_sector)[0], 15)
        self.assertLess(self.e['calculate_3tier_rs'](stock, market, None)[0], 15)

    def test_planned_risk_budget_includes_fees_and_stop_slippage(self):
        result = self.e['calculate_position_sizing'](30000000, 1, 1, 100000, 5000, 3000, 1e10)
        unit = self.e['expected_stop_risk'](100000, 95000)
        self.assertGreater(unit, 5000)
        self.assertLess(result['shares'], 60)
        self.assertLessEqual(result['shares'] * unit, 300000)
        self.assertEqual(result['normal_risk_loss'], int(np.ceil(result['shares'] * unit)))

    def test_existing_manual_positions_reduce_new_order_budget(self):
        result = self.e['calculate_position_sizing'](
            30000000, 1, 1, 100000, 5000, 3000, 1e10, remaining_account_risk=50000)
        self.assertLessEqual(result['normal_risk_loss'], 50000)
        self.assertIn('ACCOUNT_RISK_CAP', result['limiting_factor'])
        blocked = self.e['calculate_position_sizing'](
            30000000, 1, 1, 100000, 5000, 3000, 1e10, remaining_account_risk=0)
        self.assertEqual(blocked['shares'], 0)

    def test_total_and_sector_entry_risk_limits(self):
        date = self.df.index[200]
        histories = {'new': self.df, 'old': self.df.copy()}
        positions = {'old': {'shares': 10, 'entry_price': 10000, 'current_stop': 9500,
                             'sector': '반도체', 'entry_date': '2024-01-01'}}
        limit = self.e['limit_portfolio_entry'](
            1000, 'new', '반도체', 10000, 9500, 1e6, positions, histories, date,
            max_total_risk_pct=3, max_sector_risk_pct=2, max_correlated_risk_pct=10)
        self.assertLessEqual(limit['total_risk_pct'], 3)
        stop_net = self.e['floor_to_tick'](9500 * .999) * .998
        used = 10 * (histories['old'].loc[date, 'Open'] - stop_net)
        new_risk = limit['shares'] * self.e['expected_stop_risk'](10000, 9500)
        self.assertLessEqual(used + new_risk, 20000)
        self.assertEqual(limit['limiting_factor'], 'SECTOR_RISK_CAP')

    def test_correlation_limit_uses_past_returns_only(self):
        date = self.df.index[200]
        inverse = self.df.copy()
        inverse['Close'] = 30000 - inverse.Close
        positions = {'old': {'shares': 10, 'entry_price': 10000, 'current_stop': 9500,
                             'sector': '자동차', 'entry_date': '2024-01-01'}}
        args = (1000, 'new', '반도체', 10000, 9500, 1e6, positions)
        common = dict(max_total_risk_pct=10, max_sector_risk_pct=10, max_correlated_risk_pct=1)
        correlated = self.e['limit_portfolio_entry'](*args, {'new': self.df, 'old': self.df}, date, **common)
        uncorrelated = self.e['limit_portfolio_entry'](*args, {'new': self.df, 'old': inverse}, date, **common)
        self.assertLess(correlated['shares'], uncorrelated['shares'])
        altered = self.df.copy()
        altered.loc[altered.index >= date, 'Close'] = np.arange((altered.index >= date).sum()) + 1
        before = self.e['historic_correlation'](self.df, inverse, date)
        after = self.e['historic_correlation'](altered, inverse, date)
        self.assertAlmostEqual(before, after)

    def test_unavailable_correlation_is_conservatively_grouped(self):
        short = self.df.iloc[:25]
        self.assertIsNone(self.e['historic_correlation'](short, short, short.index[-1]))
        pos = {'old': {'shares': 50, 'entry_price': 10000, 'current_stop': 9000,
                       'sector': '자동차', 'entry_date': '2024-01-01'}}
        result = self.e['limit_portfolio_entry'](
            100, 'new', '반도체', 10000, 9500, 1e6, pos, {'old': short, 'new': short}, short.index[-1],
            max_total_risk_pct=10, max_sector_risk_pct=10, max_correlated_risk_pct=1)
        self.assertEqual(result['shares'], 0)

    def test_gap_exit_costs_are_reserved_in_all_three_risk_budgets(self):
        date = self.df.index[200]
        old = self.df.copy()
        old.loc[date, ['Open', 'High', 'Low', 'Close']] = [9000, 9200, 8800, 9100]
        new = self.df.copy()
        new.loc[date, ['Open', 'High', 'Low', 'Close']] = [10000, 10100, 9900, 10000]
        position = dict(shares=30, orig_shares=30, entry_price=10000, current_stop=9500,
                        sector='반도체', entry_date='2024-01-01', t1=10500, t2=11000,
                        t3=11500, t1_hit=False, t2_hit=False, t3_hit=False, holding_days=2,
                        peak_price=10000, atr=100, cash_inflow=0, total_invested=300150, track='MARKET')
        _, old_cash, old_trade = self.e['process_intraday_position'](
            position.copy(), old.loc[date], 10000, 9900, date.strftime('%Y-%m-%d'))
        self.assertEqual(old_trade['청산사유'], 'GAP_DOWN_STOP')
        gap_cost = 30 * 9000 - old_cash
        self.assertGreater(gap_cost, 0)
        cases = [(3, 10, 10, 'PORTFOLIO_RISK_CAP'), (10, 3, 10, 'SECTOR_RISK_CAP'),
                 (10, 10, 3, 'CORRELATION_RISK_CAP')]
        for total, sector, correlated, binding in cases:
            with self.subTest(binding=binding):
                result = self.e['limit_portfolio_entry'](
                    1000, 'new', '반도체', 10000, 9500, 1000000,
                    {'old': position.copy()}, {'old': old, 'new': new}, date,
                    max_total_risk_pct=total, max_sector_risk_pct=sector,
                    max_correlated_risk_pct=correlated)
                new_risk = result['shares'] * (10005 - 9490 * .998)
                self.assertLessEqual(new_risk + gap_cost, 30000)
                self.assertAlmostEqual(result['total_risk_pct'], (new_risk + gap_cost) / 1000000 * 100)
                self.assertEqual(result['limiting_factor'], binding)
                self.assertEqual(result['shares'], 54)

    def test_same_day_intraday_entry_does_not_use_the_earlier_open_as_gap_exit(self):
        date = self.df.index[200]
        old = self.df.copy()
        old.loc[date, ['Open', 'High', 'Low', 'Close']] = [9000, 12000, 8900, 11000]
        position = dict(shares=30, entry_price=10000, current_stop=9500,
                        sector='반도체', entry_date=date.strftime('%Y-%m-%d'))
        result = self.e['limit_portfolio_entry'](
            1000, 'new', '반도체', 10000, 9500, 1000000,
            {'old': position}, {'old': old, 'new': self.df}, date,
            max_total_risk_pct=3, max_sector_risk_pct=10, max_correlated_risk_pct=10)
        expected_existing = 30 * (10005 - 9490 * .998)
        expected_new = result['shares'] * (10005 - 9490 * .998)
        self.assertAlmostEqual(result['total_risk_pct'], (expected_existing + expected_new) / 1000000 * 100)
        self.assertLessEqual(expected_existing + expected_new, 30000)

    def test_profit_factor_does_not_invent_a_finite_ratio(self):
        no_losses = self.e['summarize_trades'](pd.DataFrame({'실현손익(원)': [100, 200]}))
        self.assertTrue(np.isinf(no_losses['profit_factor']))
        self.assertEqual(no_losses['profit_factor_status'], 'no_losses')
        self.assertFalse(no_losses['sample_sufficient'])
        empty = self.e['summarize_trades'](pd.DataFrame())
        self.assertTrue(np.isnan(empty['profit_factor']))
        mixed = self.e['summarize_trades'](pd.DataFrame({'실현손익(원)': [100, 200, -100]}))
        self.assertEqual(mixed['profit_factor'], 3)
        self.assertEqual(self.e['format_profit_factor'](empty['profit_factor']), '산출 불가')

    def test_wfo_small_sample_and_infinite_pf_cannot_pass(self):
        for count, pf, expected in [(2, 1.5, '표본 부족'), (30, np.inf, 'PF 추정 보류')]:
            def fake(data, reg, **kwargs):
                dates = data.loc[kwargs['eval_start_date']:].index
                ret = (1.001 ** len(dates) - 1) * 100
                return pd.DataFrame({'Total_Equity': 1}, index=dates), pd.DataFrame(), {
                    'total_trades': count, 'profit_factor': pf, 'total_return_pct': ret, 'daily_mdd': 2}
            self.e['run_daily_mtm_backtest'] = fake
            result = self.e['run_multidimensional_wfo'](self.df, self.reg)
            self.assertTrue((result['검증 기준 판정'] == expected).all())

    def test_three_modes_receive_the_same_selected_holding_period(self):
        calls = []
        def fake(*args, **kwargs):
            calls.append(kwargs['max_holding'])
            return pd.DataFrame(), pd.DataFrame(), {}
        self.e['run_daily_mtm_backtest'] = fake
        self.e['run_tricorder_backtest'](self.df, self.reg, max_holding=7)
        self.assertEqual(calls, [7, 7, 7])


if __name__ == '__main__':
    unittest.main()
