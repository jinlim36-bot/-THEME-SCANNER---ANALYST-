"""Actual-fill ledger, daily exit guidance and portable backup regressions."""
import copy
import datetime
import json
import unittest

import pandas as pd
from test_engine import load_engine


class HoldingTests(unittest.TestCase):
    def setUp(self):
        self.e = load_engine()
        self.e['market_today'] = lambda: datetime.date(2026,10,14)
        dates = pd.bdate_range(end='2026-10-14', periods=80)
        self.data = pd.DataFrame({'Open':10000.,'High':10100.,'Low':9900.,'Close':10000.,'Volume':1000000}, index=dates)
        self.position = self.e['create_holding']('005930','삼성전자',10000,100,dates[-5].date(),9500)

    def frame(self, close):
        data = self.data.copy()
        data.loc[data.index[-1], ['Open','High','Low','Close']] = [close-20,close+50,close-50,close]
        return data

    def sell(self, p, quantity, price, kind='기타', date=None):
        return self.e['record_holding_sale'](p,quantity,price,date or self.data.index[-3].date(),kind)

    def test_hold_and_partial_advice_never_invents_fills(self):
        saved = copy.deepcopy(self.position)
        self.assertEqual(self.e['holding_guidance'](self.position,self.frame(10400))['action'],'HOLD')
        for _ in range(2):
            advice = self.e['holding_guidance'](self.position,self.frame(10600))
            self.assertEqual(advice['action'],'PARTIAL')
            self.assertEqual(advice['sell_quantity'],30)
        self.assertEqual(self.position,saved)
        self.assertEqual(self.e['holding_balance'](self.position),100)

    def test_actual_partial_fills_drive_remaining_targets_and_protection(self):
        first = self.sell(self.position,30,10500,'1차 익절',self.data.index[-4].date())
        self.assertEqual(self.e['holding_balance'](first),70)
        advice = self.e['holding_guidance'](first,self.frame(10600))
        self.assertEqual(advice['action'],'HOLD')
        self.assertGreaterEqual(advice['suggested_stop'],10000)
        self.assertEqual(advice['stop'],9500)  # Advice did not apply an actual broker stop.
        advice = self.e['holding_guidance'](first,self.frame(11100))
        self.assertEqual(advice['kind'],'2차 익절')
        self.assertEqual(advice['sell_quantity'],40)
        second = self.sell(first,40,11000,'2차 익절')
        advice = self.e['holding_guidance'](second,self.frame(11200))
        self.assertEqual(advice['remaining'],30)
        self.assertGreater(advice['suggested_stop'],10500)

    def test_partial_tier_records_only_request_the_unfilled_quantity(self):
        p = self.sell(self.position,12,10500,'1차 익절')
        advice = self.e['holding_guidance'](p,self.frame(10600))
        self.assertEqual(advice['sell_quantity'],18)
        self.assertEqual(self.e['holding_balance'](p),88)
        self.assertEqual(self.e['holding_balance'](self.position),100)

    def test_stop_touch_final_target_and_duration_are_actionable(self):
        for price, label in [(9400,'손절 우선'),(11600,'3차 목표')]:
            advice = self.e['holding_guidance'](self.position,self.frame(price))
            self.assertEqual(advice['action'],'EXIT')
            self.assertIn(label,advice['label'])
            self.assertEqual(advice['sell_quantity'],100)
        data = self.frame(10400)
        data.loc[data.index[-1],'Low'] = 9400
        self.assertEqual(self.e['holding_guidance'](self.position,data)['action'],'REVIEW')
        old = self.e['create_holding']('005930','삼성전자',10000,100,self.data.index[-15].date(),9500)
        self.assertIn('보유 기한',self.e['holding_guidance'](old,self.frame(10400))['label'])

    def test_trend_break_after_actual_first_exit_has_priority(self):
        p = self.sell(self.position,30,10500,'1차 익절',self.data.index[-4].date())
        data = self.frame(10200)
        data.loc[data.index[-4:-1],['Open','High','Low','Close']] = [10400,10500,10300,10400]
        advice = self.e['holding_guidance'](p,data)
        self.assertEqual(advice['action'],'EXIT')
        self.assertIn('추세 이탈',advice['label'])
        self.assertEqual(advice['sell_quantity'],70)

    def test_recorded_stop_never_decreases_and_budget_uses_actual_remaining(self):
        before = self.e['registered_holding_risk']([self.position])
        p = self.sell(self.position,30,10500,'1차 익절')
        self.assertLess(self.e['registered_holding_risk']([p]),before)
        raised = self.e['record_holding_stop'](p,10000,self.data.index[-2].date())
        self.assertLess(self.e['registered_holding_risk']([raised]),self.e['registered_holding_risk']([p]))
        with self.assertRaises(ValueError):
            self.e['record_holding_stop'](raised,9900,self.data.index[-1].date())
        self.assertEqual(p['stop_updates'],[])

    def test_overselling_backdating_and_invalid_exit_tiers_are_rejected(self):
        for quantity,price,kind,date in [(101,10500,'기타','2026-10-14'),(30,10500,'전량 청산','2026-10-14'),
                                       (30,9900,'1차 익절','2026-10-14'),(40,11000,'2차 익절','2026-10-14'),
                                       (30,10500,'기타','2024-01-01')]:
            with self.subTest(quantity=quantity,kind=kind,date=date), self.assertRaises(ValueError):
                self.sell(self.position,quantity,price,kind,date)
        p = self.sell(self.position,10,10500,date='2026-10-14')
        with self.assertRaises(ValueError):
            self.sell(p,10,10500,date='2026-10-13')

    def test_closed_ledger_reports_realized_costs_without_loading_quotes(self):
        p = self.sell(self.position,100,11000,'전량 청산')
        advice = self.e['holding_guidance'](p,None)
        self.assertEqual(advice['action'],'CLOSED')
        self.assertEqual(advice['remaining'],0)
        self.assertEqual(advice['realized_pnl'],97300)
        self.assertEqual(self.e['registered_holding_risk']([p]),0)

    def test_quote_older_than_actual_fill_never_produces_false_advice(self):
        p = self.sell(self.position,30,10500,'1차 익절',self.data.index[-1].date())
        advice = self.e['holding_guidance'](p,self.frame(10600).iloc[:-1])
        self.assertEqual(advice['action'],'DATA')
        self.assertEqual(advice['remaining'],70)
        self.e['market_today'] = lambda: datetime.date(2026,11,1)
        self.assertEqual(self.e['holding_guidance'](p,self.frame(10600))['action'],'DATA')

    def test_backup_roundtrip_handles_partial_closed_and_stop_history(self):
        p = self.e['record_holding_stop'](self.position,10000,self.data.index[-4].date())
        p = self.sell(p,30,10500,'1차 익절',self.data.index[-3].date())
        p = self.sell(p,70,11000,'전량 청산')
        encoded = self.e['export_holding_book']([p])
        self.assertEqual(self.e['import_holding_book'](encoded),[p])
        self.assertEqual(self.e['merge_holding_books']([p],[p]),[p])
        conflict = dict(p,name='다른 기록')
        with self.assertRaises(ValueError):
            self.e['merge_holding_books']([p],[conflict])

    def test_backup_validation_is_atomic_and_rejects_invalid_economic_records(self):
        good = self.e['export_holding_book']([self.position])
        for change in ['quantity','duplicate','nan','overfill','missing_id','future']:
            root = json.loads(good)
            p = root['positions'][0]
            if change == 'quantity': p['quantity'] = -1
            elif change == 'duplicate': root['positions'].append(dict(p))
            elif change == 'nan': p['entry_price'] = float('nan')
            elif change == 'overfill': p['fills'] = [{'id':'sale1','quantity':101,'price':11000,'date':'2026-10-14','kind':'기타'}]
            elif change == 'missing_id': p['id'] = ''
            elif change == 'future': p['entry_date'] = '2099-01-01'
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.e['import_holding_book'](json.dumps(root).encode())
        self.assertEqual(self.position['fills'],[])
        with self.assertRaises(ValueError): self.e['import_holding_book'](b'not-json')
        with self.assertRaises(ValueError): self.e['import_holding_book'](b' '*(1024*1024+1))

    def test_small_positions_can_really_close_instead_of_leaving_zero_share_positions(self):
        self.assertEqual(self.e['partial_exit_quantities'](1),(1,0))
        self.assertEqual(self.e['partial_exit_quantities'](2),(1,1))
        self.assertEqual(self.e['partial_exit_quantities'](100),(30,40))
        p = dict(entry_price=10000,current_stop=9500,shares=1,orig_shares=1,
                 t1=10500,t2=11000,t3=11500,t1_hit=False,t2_hit=False,t3_hit=False,
                 holding_days=0,peak_price=10000,atr=100,cash_inflow=0,total_invested=10005,
                 entry_date='2026-10-13',track='REBOUND')
        active, cash, trade = self.e['process_intraday_position'](p,dict(Open=10400,High=10600,Low=10300,Close=10550),10000,9900,'2026-10-14')
        self.assertIsNone(active)
        self.assertEqual(trade['청산사유'],'TARGET_1R_FULL')
        self.assertGreater(cash,0)


if __name__ == '__main__':
    unittest.main()
