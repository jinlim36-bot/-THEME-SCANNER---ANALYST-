"""Offline equivalence and data-loading regressions for the optimization path."""
import copy
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_engine import history, load_engine


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.e = load_engine()
        self.df = history(180)
        self.reg = pd.DataFrame({'Regime': 'FAVORABLE', 'KS_Close': self.df.Close}, index=self.df.index)

    def assert_backtests_equal(self, left, right):
        pd.testing.assert_frame_equal(left[0], right[0], check_exact=True)
        pd.testing.assert_frame_equal(left[1], right[1], check_exact=True)
        pd.testing.assert_series_equal(pd.Series(left[2]), pd.Series(right[2]), check_exact=True)

    def test_shared_features_preserve_daily_equity_trades_and_risk_metrics(self):
        cache = {}
        supply = pd.Series(np.where(np.arange(len(self.df)) % 3, 10, -10), index=self.df.index)
        reg = self.reg.copy()
        reg.loc[reg.index[80:100], 'Regime'] = 'NEUTRAL'
        reg.loc[reg.index[100:110], 'Regime'] = 'RISK-OFF'
        reg.loc[reg.index[110:115], 'Regime'] = 'UNKNOWN'
        reg.loc[reg.index[150], 'KS_Close'] = np.nan
        original = self.e['evaluate_execution_features']
        computed_dates = []
        def observe(data, *args, **kwargs):
            computed_dates.append(data.index[-1])
            return original(data, *args, **kwargs)
        for score, trail, hold, mode in [(70, 1.2, 7, 'CONSERVATIVE'),
                                          (75, 1.5, 10, 'NEUTRAL'),
                                          (80, 2.0, 14, 'OPTIMISTIC')]:
            with self.subTest(score=score, mode=mode):
                args = dict(min_score=score, trail_atr_mult=trail, max_holding=hold,
                            execution_mode=mode, historical_supply_series=supply,
                            sector_slice=self.df[['Close']], eval_start_date=self.df.index[70])
                self.e['evaluate_execution_features'] = original
                reference = self.e['run_daily_mtm_backtest'](self.df, reg, **args)
                self.e['evaluate_execution_features'] = observe
                optimized = self.e['run_daily_mtm_backtest'](self.df, reg, _feature_cache=cache, **args)
                self.assert_backtests_equal(reference, optimized)
        self.assertEqual(len(computed_dates), len(set(computed_dates)))
        self.assertGreater(len(cache), 0)

    def test_wfo_keeps_training_caches_separate_and_oos_uncached(self):
        calls = []
        def fake(data, reg, **kwargs):
            start = kwargs['eval_start_date']
            calls.append((data.index[0], start, kwargs.get('_feature_cache')))
            dates = data.loc[start:].index
            return pd.DataFrame({'Total_Equity': 1}, index=dates), pd.DataFrame(), {
                'total_trades': 0, 'profit_factor': np.nan, 'total_return_pct': 0, 'daily_mdd': 0}
        self.e['run_daily_mtm_backtest'] = fake
        self.e['run_multidimensional_wfo'](history(420), self.reg)
        self.assertEqual(len(calls), 4 * 28)  # No redundant fallback backtest.
        previous_cache = None
        for offset in range(0, len(calls), 28):
            cache = calls[offset][2]
            self.assertIsNotNone(cache)
            self.assertIsNot(cache, previous_cache)
            self.assertTrue(all(call[2] is cache for call in calls[offset:offset+27]))
            self.assertIsNone(calls[offset+27][2])
            previous_cache = cache

    def test_cached_features_use_only_prior_signal_dates_and_do_not_mutate(self):
        prefix = self.df.iloc[:100]
        cache = {}
        result = self.e['run_daily_mtm_backtest'](prefix, self.reg, _feature_cache=cache,
                                                 eval_start_date=prefix.index[80])
        self.assertEqual(min(cache), prefix.index[79])
        self.assertLessEqual(max(cache), prefix.index[-1])
        saved = copy.deepcopy(cache)
        changed_future = self.df.copy()
        changed_future.loc[changed_future.index >= self.df.index[100], 'Close'] = 1e9
        again = self.e['run_daily_mtm_backtest'](changed_future.iloc[:100], self.reg,
                                                _feature_cache=cache, eval_start_date=prefix.index[80],
                                                min_score=80, max_holding=14)
        self.assertEqual({date: cache[date] for date in saved}, saved)
        direct = self.e['run_daily_mtm_backtest'](prefix, self.reg, eval_start_date=prefix.index[80],
                                                 min_score=80, max_holding=14)
        self.assert_backtests_equal(again, direct)
        self.assertFalse(result[0].empty)

    def supply(self, tables, status=200):
        self.e['io'] = io
        self.e['requests'] = SimpleNamespace(get=lambda *a, **k: SimpleNamespace(status_code=status, content=b'fixture'))
        with patch.object(pd, 'read_html', return_value=tables):
            return self.e['get_supply_data']('005930')

    def test_foreign_flows_do_not_require_unused_institutional_data(self):
        irrelevant = pd.DataFrame({'외국인 보유율': [1]})
        foreign = pd.DataFrame({'외국인_순매매량': ['1,000', '-200', '300', '400', '500']})
        result = self.supply([irrelevant, foreign])
        self.assertEqual(result, {'status': 'OK', 'frgn_5': 2000})

    def test_incomplete_supply_and_http_failures_remain_unverified(self):
        for tables, status in [([pd.DataFrame({'외국인_순매매량': [1, 2, 3, 4]})], 200), ([], 403)]:
            with self.subTest(status=status):
                self.assertEqual(self.supply(tables, status), {'status': 'FAIL', 'frgn_5': None})

    def test_sector_lookup_uses_master_then_fallback_without_per_stock_fetch(self):
        master = pd.DataFrame({'Code': ['005930', '000660'], 'Sector_Raw': ['반도체 제조', None]})
        listing = pd.DataFrame({'Code': ['005930', '000660', '123456'], 'Sector': ['자동차', '반도체', '의약품']})
        requests = []
        self.e['load_all_krx_master'] = lambda: requests.append(1) or master
        lookup = self.e['build_sector_lookup'](listing)
        self.assertEqual(self.e['get_sector_for_stock']('005930', sector_lookup=lookup), '반도체')
        self.assertEqual(self.e['get_sector_for_stock']('000660', sector_lookup=lookup), '반도체')
        self.assertEqual(self.e['get_sector_for_stock']('123456', sector_lookup=lookup), '바이오')
        self.assertEqual(self.e['get_sector_for_stock']('999999', sector_lookup=lookup), '대표지수')
        self.assertEqual(len(requests), 1)

    def test_sector_series_follows_the_price_loader_snapshot(self):
        snapshots = [self.df[['Close']].iloc[:-1], self.df[['Close']]]
        self.e['load_price_history'] = lambda *args: snapshots.pop(0)
        old = self.e['load_sector_etf_series']('반도체')
        new = self.e['load_sector_etf_series']('반도체')
        self.assertEqual(old.index[-1], self.df.index[-2])
        self.assertEqual(new.index[-1], self.df.index[-1])


if __name__ == '__main__':
    unittest.main()
