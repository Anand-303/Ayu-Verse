import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, session
from app import app
import unittest

class DoshaTestRoutes(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()
        self.ctx = app.app_context()
        self.ctx.push()

    def tearDown(self):
        self.ctx.pop()

    def test_dosha_test_route(self):
        # Test accessing dosha test without login
        response = self.client.get('/dosha-test', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Please log in', response.data)

    def test_process_dosha_test_route(self):
        # Test processing dosha test without login
        response = self.client.post('/process-dosha-test', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Please log in', response.data)

if __name__ == '__main__':
    unittest.main()
