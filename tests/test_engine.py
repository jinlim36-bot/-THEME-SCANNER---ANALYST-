"""Offline regressions for financial logic; do not launch Streamlit or fetch data."""
import ast
import datetime
import logging
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


def load_engine():
    source = (Path(__file__).resolve().parents[1] / 'app.py').read_text()
    tree = ast.parse(source)
    constants = {'KOREA_TICK_TABLE', 'SECTOR_ETF_MAP', 'KRX_INDUSTRY_KEYWORD_MAP'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             or (isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets))]
    for node in nodes:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    scope = dict(pd=pd, np=np, math=math, datetime=datetime,
                 logger=logging.getLogger('engine-test'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'app.py', 'exec'), scope)
    return scope


def history(size=420):
    dates = pd.bdate_range('2024-01-01', periods=size)
    close = 10000 + np.arange(size) * 5 + np.sin(np.arange(size) / 4) * 100
    return pd.DataFrame({'Open': close - 15, 'High': close + 80,
                         'Low': close - 80, 'Close': close,
                         'Volume': np.full(size, 1000000)}, index=dates)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.e = load_engine()
        self.df = history()
        self.reg = pd.DataFrame({'Regime': 'FAVORABLE', 'KS_Close': self.df.Close},
                                index=self.df.index)

    def test_precomputed_indicators_are_causal_and_equivalent(self):
        prepared = self.e['prepare_engine_history'](self.df)
        for end in [60, 100, 200]:
            prefix = self.df.iloc[:end]
            local = self.e['prepare_engine_history'](prefix)
            columns = [c for c in local if c.startswith('_engine_')]
            pd.testing.assert_frame_equal(prepared.iloc[:end][columns], local[columns])
            self.assertAlmostEqual(self.e['calculate_wilder_atr'](prefix),
                                   self.e['calculate_wilder_atr'](prepared.iloc[:end]))
            plain = self.e['calculate_100p_score'](prefix)
            fast = self.e['calculate_100p_score'](prepared.iloc[:end])
            for key in plain:
                if isinstance(plain[key], float):
                    self.assertAlmostEqual(plain[key], fast[key])
                else:
                    self.assertEqual(plain[key], fast[key])

    def test_portfolio_uses_historic_supply_and_preserves_cash(self):
        data = self.df.copy()
        data.index = pd.bdate_range(end=datetime.date.today(), periods=len(data))
        reg = pd.DataFrame({'Regime': 'FAVORABLE', 'KS_Close': data.Close}, index=data.index)
        self.e['load_price_history'] = lambda *args: data.copy()
        self.e['load_historical_foreign_series'] = lambda *args: pd.Series(10, index=data.index)
        self.e['get_sector_for_stock'] = lambda *args: '대표지수'
        self.e['load_sector_etf_series'] = lambda *args: data[['Close']].copy()
        original = self.e['evaluate_full_execution_pipeline']
        flows = []
        def observe(*args, **kwargs):
            flows.append(kwargs['has_frgn'])
            return original(*args, **kwargs)
        self.e['evaluate_full_execution_pipeline'] = observe
        daily, trades, metrics = self.e['run_portfolio_backtest'](['005930', '000660'], reg)
        self.assertTrue(any(flows))
        self.assertFalse(daily.empty)
        self.assertGreater(len(trades), 0)
        self.assertGreaterEqual(daily.Cash.min(), 0)
        self.assertLessEqual(daily.Num_Positions.max(), 3)
        self.assertEqual(metrics['total_trades'], len(trades))
        self.assertLessEqual(metrics['max_entry_stop_risk_pct'], 3.0 + 1e-9)
        self.assertTrue((daily.New_Entry_Stop_Risk_pct.dropna() <= 3.0 + 1e-9).all())

    def test_tick_boundaries(self):
        tick = self.e['get_tick_size']
        for price, expected in [(1999, 1), (2000, 5), (5000, 10), (20000, 50),
                                (50000, 100), (200000, 500), (500000, 1000)]:
            self.assertEqual(tick(price), expected)

    def test_stop_is_nearest_with_noise_buffer(self):
        self.e['calculate_wilder_atr'] = lambda *args: 100
        data = pd.DataFrame({'Close': [10000] * 30, 'Low': [9900] * 30,
                             'High': [10100] * 30})
        levels = self.e['calculate_execution_levels'](data, 10000)
        self.assertEqual(levels['stop_price'], 9880)
        self.assertEqual(levels['risk_1r'], 120)
        self.assertEqual(levels['target_1r'], 10120)

    def test_rs_excludes_future_data_and_aligns_dates(self):
        data = self.df.iloc[:60]
        market = self.df[['Close']].copy()
        sector = market.copy()
        sector.loc[sector.index > data.index[-1], 'Close'] = 1e9
        score, _ = self.e['calculate_3tier_rs'](data, market, sector)
        self.assertEqual(score, 10)
        score2, _ = self.e['calculate_3tier_rs'](data, market.iloc[:10], sector)
        self.assertEqual(score2, 0)

    def test_evaluation_dates_and_empty_range(self):
        start = self.df.index[100]
        daily, trades, metrics = self.e['run_daily_mtm_backtest'](
            self.df, self.reg, eval_start_date=start)
        self.assertEqual(daily.index[0], start)
        self.assertEqual(len(daily), len(self.df) - 100)
        self.assertGreaterEqual(daily.Cash.min(), 0)
        self.assertGreaterEqual(metrics['daily_mdd'], 0)
        empty = self.e['run_daily_mtm_backtest'](
            self.df, self.reg, eval_start_date=self.df.index[-1] + pd.Timedelta(days=1))
        self.assertTrue(empty[0].empty)

    def test_first_evaluation_day_can_execute_previous_close_signal(self):
        original = self.e['evaluate_full_execution_pipeline']
        seen = []
        def signal(data, *args, **kwargs):
            seen.append(data.index[-1])
            decision = original(data, *args, **kwargs)
            decision.update(action='BUY', multiplier=1.0)
            decision['scored']['track'] = 'MARKET'
            return decision
        self.e['evaluate_full_execution_pipeline'] = signal
        start = self.df.index[100]
        daily, _, _ = self.e['run_daily_mtm_backtest'](
            self.df.iloc[:110], self.reg, eval_start_date=start)
        self.assertEqual(seen[0], self.df.index[99])
        self.assertGreater(daily.Stock_Value.iloc[0], 0)

    def test_first_day_loss_counts_in_mdd(self):
        data = self.df.iloc[:101].copy()
        data.iloc[100, data.columns.get_loc('Low')] = 9000
        original = self.e['evaluate_full_execution_pipeline']
        def signal(*args, **kwargs):
            decision = original(*args, **kwargs)
            decision.update(action='BUY', multiplier=1.0)
            decision['scored']['track'] = 'MARKET'
            return decision
        self.e['evaluate_full_execution_pipeline'] = signal
        daily, trades, metrics = self.e['run_daily_mtm_backtest'](
            data, self.reg, eval_start_date=data.index[100])
        self.assertEqual(len(daily), 1)
        self.assertEqual(len(trades), 1)
        self.assertLess(daily.Total_Equity.iloc[0], 30000000)
        self.assertAlmostEqual(metrics['daily_mdd'], abs(metrics['total_return_pct']))

    def test_breakeven_ambiguity_respects_execution_mode(self):
        position = dict(entry_price=10000, current_stop=9500, shares=100, orig_shares=100,
                        t1=10500, t2=11000, t3=11500, t1_hit=False, t2_hit=False,
                        t3_hit=False, holding_days=1, peak_price=10000, atr=100,
                        cash_inflow=0, total_invested=1000000, entry_date='2024-01-01',
                        track='MARKET')
        bar = dict(Open=10100, High=10600, Low=9990, Close=10500)
        conservative, _, trade = self.e['process_intraday_position'](
            position.copy(), bar, 10000, 9900, '2024-01-02', execution_mode='CONSERVATIVE')
        optimistic, _, other = self.e['process_intraday_position'](
            position.copy(), bar, 10000, 9900, '2024-01-02', execution_mode='OPTIMISTIC')
        self.assertIsNone(conservative)
        self.assertEqual(trade['청산사유'], 'T1_THEN_BREAKEVEN')
        self.assertIsNotNone(optimistic)
        self.assertIsNone(other)

    def test_wfo_windows_parameters_and_daily_normalization(self):
        calls = []
        def fake(data, reg, **kwargs):
            start = kwargs['eval_start_date']
            days = data.loc[start:].index
            calls.append((data.index[0], start, data.index[-1], kwargs))
            # Equal daily returns should yield 100% WFE despite differing durations.
            rate = (1.001 ** len(days) - 1) * 100
            return (pd.DataFrame({'Total_Equity': 1}, index=days), pd.DataFrame(),
                    {'total_trades': 30, 'profit_factor': 1.5, 'total_return_pct': rate, 'daily_mdd': 2.0})
        self.e['run_daily_mtm_backtest'] = fake
        result = self.e['run_multidimensional_wfo'](self.df, self.reg)
        self.assertEqual(len(result), 4)
        self.assertEqual(len(calls), 4 * 28)
        self.assertTrue(all(value == '100.0%' for value in result['WFE(%)']))
        for offset in range(0, len(calls), 28):
            is_calls = calls[offset:offset+27]
            oos = calls[offset+27]
            self.assertEqual(len({(c[3]['min_score'], c[3]['trail_atr_mult'],
                                  c[3]['max_holding']) for c in is_calls}), 27)
            self.assertLess(is_calls[0][2], oos[1])
            self.assertLess(oos[0], oos[1])
        self.assertEqual(calls[-1][2], self.df.index[-1])

    def test_trailing_multiplier_and_cash_accounting(self):
        base = dict(entry_price=10000, current_stop=9500, shares=30, orig_shares=100,
                    t1=10500, t2=11000, t3=12000, t1_hit=True, t2_hit=True,
                    t3_hit=False, holding_days=1, peak_price=11000, atr=100,
                    cash_inflow=700000, total_invested=1000000, entry_date='2024-01-01',
                    track='MARKET')
        bar = {'Open':11000,'High':11200,'Low':11000,'Close':11100}
        tight, _, _ = self.e['process_intraday_position'](
            base.copy(), bar, 10000, 10000, '2024-01-02', trail_atr_mult=1.2)
        wide, _, _ = self.e['process_intraday_position'](
            base.copy(), bar, 10000, 10000, '2024-01-02', trail_atr_mult=2.0)
        self.assertGreater(tight['current_stop'], wide['current_stop'])
        bar = dict(Open=9400, High=9500, Low=9300, Close=9400)
        closed, gain, trade = self.e['process_intraday_position'](
            base.copy(), bar, 10000, 10000, '2024-01-02')
        self.assertIsNone(closed)
        self.assertAlmostEqual(trade['실현손익(원)'], int(700000 + gain - 1000000))


if __name__ == '__main__':
    unittest.main()
