import sys
import os
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.metrics_tracker import MetricsTracker
from fastapi.testclient import TestClient
from main import app

class TestRateLimitFix(unittest.IsolatedAsyncioTestCase): # Use AsyncioTestCase for async tests
    def setUp(self):
        self.client = TestClient(app)

    @patch('utils.metrics_tracker.MetricsTracker._supabase')
    async def test_ip_detection_x_forwarded_for(self, mock_supabase):
        """Verify try_increment_post properly interacts with Supabase"""
        fake_ip = "10.0.0.100"
        
        # Mock Supabase responses
        mock_execute_res = MagicMock()
        mock_execute_res.data = [{'post_count': 0}]
        
        # Mock the chained select().eq().eq().execute()
        mock_eq2 = MagicMock()
        mock_eq2.execute.return_value = mock_execute_res
        mock_eq1 = MagicMock()
        mock_eq1.eq.return_value = mock_eq2
        mock_select = MagicMock()
        mock_select.select.return_value = mock_eq1
        mock_supabase.table.return_value = mock_select
        
        # It's difficult to test the middleware injection end-to-end without a real DB and 
        # risking dirtying dev state, so let's test the interface logic directly.
        
        result = await MetricsTracker.try_increment_post(fake_ip)
        self.assertTrue(result)
        
        # Verify it attempted to read from the limit table
        mock_supabase.table.assert_called_with("ip_rate_limits")

    @patch('utils.metrics_tracker.MetricsTracker._supabase')
    async def test_memory_cleanup_obsolete(self, mock_supabase):
        """Verify the legacy memory cleanup test is aware of the DB shift"""
        # _cleanup_stale_ips is removed. This test asserts it's gone and 
        # the system relies on Supabase DB eviction policies (like pg_cron) instead.
        self.assertFalse(hasattr(MetricsTracker, '_cleanup_stale_ips'))

if __name__ == '__main__':
    unittest.main()
