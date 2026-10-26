"""KRX date-bootstrap failure must not prevent reading FDR's published listings."""
import datetime
import io
from types import SimpleNamespace
import unittest
import pandas as pd
from test_engine import load_engine


class ListingTests(unittest.TestCase):
    def setUp(self):
        self.e = load_engine()
        self.e['io'] = io
        self.e['market_today'] = lambda: datetime.date(2026,10,26)
        self.warnings = []
        self.e['st'] = SimpleNamespace(warning=self.warnings.append)
        self.table = pd.DataFrame({'Code':['005930','000660','123456'], 'Name':['삼성전자','SK하이닉스','코스닥종목'],
             'MarketId':['STK','STK','KSQ'], 'Close':[200000,100000,10000], 'Volume':[100,200,300],
             'Amount':[20000000,20000000,3000000], 'Marcap':[1e12,1e12,1e10]})
        self.content = self.table.to_csv(index=False).encode()

    def response(self, status=200, content=None):
        def check():
            if status >= 400: raise RuntimeError('HTTP failure')
        return SimpleNamespace(status_code=status,content=self.content if content is None else content,raise_for_status=check)

    def test_market_filter_preserves_codes_and_never_calls_krx_bootstrap(self):
        self.e['requests'] = SimpleNamespace(get=lambda *a, **k:self.response())
        self.e['fdr'] = SimpleNamespace(StockListing=lambda *a: self.fail('KRX bootstrap called'))
        result = self.e['load_stock_listing']('KOSPI')
        self.assertEqual(result.Code.tolist(),['005930','000660'])
        self.assertEqual(result.attrs['listing_date'],'2026-10-26')
        self.assertEqual(len(self.e['load_stock_listing']('KRX')),3)
        self.assertEqual(self.warnings,[])

    def test_unpublished_day_falls_back_to_friday_with_date_visible(self):
        calls=[]
        def get(url, **kwargs):
            calls.append(url)
            return self.response(404 if '2026-10-26' in url else 200)
        self.e['requests'] = SimpleNamespace(get=get)
        result = self.e['load_stock_listing']('KOSPI')
        self.assertEqual(result.attrs['listing_date'],'2026-10-23')
        self.assertEqual(len(calls),2)

    def test_missing_or_blocked_recent_sources_fail_without_old_or_fake_data(self):
        for status in [404,403]:
            with self.subTest(status=status):
                self.e['requests'] = SimpleNamespace(get=lambda *a, **k:self.response(status))
                self.assertTrue(self.e['load_stock_listing']('KOSPI').empty)
        self.assertTrue(self.warnings)

    def test_new_alphanumeric_krx_codes_do_not_break_the_entire_listing(self):
        table=self.table.copy(); table.loc[0,'Code']='0126Z0'
        result=self.e['normalize_published_listing'](table.to_csv(index=False).encode())
        self.assertEqual(result.Code.iloc[0],'0126Z0')

    def test_invalid_schema_values_and_market_codes_are_rejected(self):
        for column,value in [('Close',-1),('Marcap',None),('Code','wrong'),('MarketId','WRONG')]:
            data=self.table.copy();data.loc[0,column]=value
            with self.subTest(column=column),self.assertRaises(ValueError):
                self.e['normalize_published_listing'](data.to_csv(index=False).encode())
        with self.assertRaises(ValueError):self.e['normalize_published_listing'](b'<html>error</html>')


if __name__=='__main__':unittest.main()
