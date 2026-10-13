"""Exercise the OpenAI integration without paid requests or real credentials."""
from contextlib import nullcontext
import logging
import re
from types import SimpleNamespace
import unittest

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError
from test_engine import load_engine


class AIBriefingTests(unittest.TestCase):
    def setUp(self):
        self.e = load_engine()
        self.messages = []
        st = SimpleNamespace(caption=self.messages.append, warning=self.messages.append,
                             text=self.messages.append, expander=lambda *a, **kw: nullcontext())
        self.e.update(st=st, re=re, httpx=httpx, APIStatusError=APIStatusError,
                      APIConnectionError=APIConnectionError, APITimeoutError=APITimeoutError,
                      active_key='sk-test-secret', active_model='gpt-4.1-mini',
                      logger=logging.getLogger('ai-test'))
        self.calls = []

    def client(self, result):
        def create(**kwargs):
            self.calls.append(kwargs)
            if isinstance(result, Exception):
                raise result
            return result
        return SimpleNamespace(responses=SimpleNamespace(create=create))

    def test_success_uses_responses_api_and_does_not_store(self):
        result = SimpleNamespace(output_text='리스크 설명입니다.', status='completed')
        text = self.e['generate_ai_briefing'](self.client(result), 'quantitative prompt')
        self.assertEqual(text, result.output_text)
        self.assertEqual(self.calls, [dict(model='gpt-4.1-mini', input='quantitative prompt',
                                         max_output_tokens=800, store=False)])

    def test_quota_error_is_reported_and_key_redacted_without_retry(self):
        response = httpx.Response(429, request=httpx.Request('POST', 'https://api.openai.com/v1/responses'))
        error = APIStatusError('insufficient_quota key=sk-test-secret', response=response,
                               body={'code': 'insufficient_quota'})
        text = self.e['generate_ai_briefing'](self.client(error), 'prompt')
        self.assertIn('429', text)
        self.assertIn('AI-DIAG-3', text)
        self.assertIn('API 잔액', text)
        self.assertNotIn('sk-test-secret', text)
        self.assertTrue(all('sk-test-secret' not in m for m in self.messages))
        self.assertEqual(len(self.calls), 1)

    def test_timeout_and_proxy_connection_are_distinguished(self):
        request = httpx.Request('POST', 'https://api.openai.com/v1/responses')
        timeout = APITimeoutError(request=request)
        text = self.e['generate_ai_briefing'](self.client(timeout), 'prompt')
        self.assertIn('Timeout', text)
        connection = APIConnectionError(request=request)
        connection.__cause__ = httpx.ProxyError('blocked')
        text = self.e['generate_ai_briefing'](self.client(connection), 'prompt')
        self.assertIn('ProxyError', text)

    def test_incomplete_text_is_not_presented_as_success(self):
        result = SimpleNamespace(output_text='partial', status='incomplete',
                                 incomplete_details=SimpleNamespace(reason='max_output_tokens'), output=[])
        text = self.e['generate_ai_briefing'](self.client(result), 'prompt')
        self.assertIn('max_output_tokens', text)
        self.assertNotEqual(text, 'partial')

    def test_refusal_and_empty_response_are_diagnosed(self):
        result = SimpleNamespace(output_text='', status='completed', incomplete_details=None,
                                 output=[SimpleNamespace(content=[SimpleNamespace(type='refusal')])])
        text = self.e['generate_ai_briefing'](self.client(result), 'prompt')
        self.assertIn('거절', text)
        text = self.e['generate_ai_briefing'](self.client(None), 'prompt')
        self.assertIn('응답 없음', text)


if __name__ == '__main__':
    unittest.main()
