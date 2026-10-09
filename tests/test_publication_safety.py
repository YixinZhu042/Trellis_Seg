"""Offline guards for the public source release."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


class PublicationSafetyTests(unittest.TestCase):
    def test_api_credentials_are_not_embedded(self):
        spec=importlib.util.spec_from_file_location('public_seg_api',ROOT/'scripts/segment_views_via_api.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict('os.environ',{},clear=True):spec.loader.exec_module(module)
        self.assertEqual(module.API_KEY,'')
        with patch.object(module.requests,'post') as post:
            with self.assertRaisesRegex(ValueError,'SEG_API_KEY'):module.call_api(None)
            post.assert_not_called()

    def test_unconfigured_remote_api_fails_without_network(self):
        from scripts import train_seg_vae_v2_remote as remote
        with patch.object(remote,'BOARD',''):
            with self.assertRaisesRegex(ValueError,'TRELLIS_REMOTE_BOARD'):
                remote.api_client(Path('/nonexistent/sdk'),'master')

    def test_remote_worker_forwards_only_required_non_secret_settings(self):
        from scripts import train_seg_vae_v2_remote as remote
        with patch.dict('os.environ',{'SEG_API_KEY':'not-to-forward','HF_TOKEN':'not-to-forward'}):
            args=remote.remote_worker_prefix()
        self.assertEqual(args[0],'env')
        keys={x.split('=',1)[0] for x in args if '=' in x}
        self.assertEqual(keys,{'TRELLIS_REMOTE_API_ROOT','TRELLIS_PYTHON','TRELLIS_REMOTE_NODES','TRELLIS_REMOTE_BOARD'})
        self.assertNotIn('not-to-forward',' '.join(args))


if __name__=='__main__':unittest.main()
