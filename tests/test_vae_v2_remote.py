import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

from scripts.train_seg_vae_v2_remote import dispatch, node_command, read_probe


class RemoteLauncherTests(unittest.TestCase):
    def test_command_is_quoted_persistent_and_no_resume(self):
        job = {'rank': 2, 'log': '/tmp/a b.log', 'exit_file': '/tmp/a b.exit'}
        command = node_command(job, Path('/tmp/config x.json'), Path('/tmp/output x'),
                               Path('/tmp/log x'), Path('/tmp/cache x'), '/bin/python',
                               ['10.0.0.1', '10.0.0.2', '10.0.0.3'], 32123, 5000)
        subprocess.run(['bash', '-n', '-c', command], check=True)
        body = command
        self.assertIn('--nproc-per-node 8', body)
        self.assertIn('--node-rank 2', body)
        self.assertIn('--master-port 32123', body)
        self.assertIn('TRITON_CACHE_DIR=/tmp/cache x/node2/triton', body)
        train_args = shlex.split(command)
        self.assertEqual(train_args[train_args.index('--config')+1], '/tmp/config x.json')
        self.assertNotIn('--resume', body)
        self.assertNotIn('nohup', body)
        self.assertNotIn('taiji', body)
        self.assertNotIn('ssh ', body)
        self.assertNotIn('$(', command)
        self.assertNotIn('`', command)
        self.assertNotIn('bash -c', command)

    def test_preflight_rejects_failed_or_missing_marker(self):
        with self.assertRaises(RuntimeError):
            read_probe({'exit_code': 1, 'output': 'bad'})
        with self.assertRaises(RuntimeError):
            read_probe({'exit_code': 0, 'output': 'not a result'})
        self.assertEqual(read_probe({'exit_code': 0, 'output': 'noise\nVAE_PREFLIGHT={"gpus":8}\n'}), {'gpus': 8})

    def test_resume_from_command(self):
        command=node_command({'rank':0,'log':'/tmp/x.log','exit_file':'/tmp/x.exit'},
                             Path('/tmp/a.json'),Path('/tmp/out'),Path('/tmp/log'),Path('/tmp/cache'),
                             '/bin/python',['10.0.0.1','10.0.0.2','10.0.0.3'],32123,10000,
                             resume_from=Path('/tmp/source state.pt'),manifest_workers=4)
        args=shlex.split(command)
        self.assertEqual(args[args.index('--resume_from')+1],'/tmp/source state.pt')
        self.assertIn('--manifest_workers 4',command)
        self.assertNotIn('--resume',args)

    def test_immediate_dispatch_and_scope_cleanup(self):
        class FakeClient:
            def __init__(self): self.events = []
            def create_session(self, ip, name):
                self.events.append(('create', ip))
                return {'id': name, 'write_guard': True}
            def exec_command(self, sid, command, timeout):
                self.events.append(('exec', sid))
                # The ID must already be durable if submission raises.
                self.assert_saved(sid)
                if sid == 'owned1': raise RuntimeError('submission failed')
                return {'exit_code': 0}
            def delete_session(self, sid):
                self.events.append(('delete', sid))
                return {'deleted': sid}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'plan.json'
            client = FakeClient()
            client.assert_saved = lambda sid: self.assertIn(sid, [j.get('session_id') for j in json.loads(path.read_text())['jobs']])
            plan = {'jobs': [{'ip': f'node{i}', 'rank': i, 'session_name': f'owned{i}', 'command': 'true'} for i in range(3)]}
            with self.assertRaisesRegex(RuntimeError, 'submission failed'):
                dispatch(client, plan, path)
            self.assertEqual(client.events, [('create', 'node0'), ('exec', 'owned0'),
                                              ('create', 'node1'), ('exec', 'owned1'),
                                              ('delete', 'owned0'), ('delete', 'owned1')])
            self.assertEqual(json.loads(path.read_text())['status'], 'submission_failed')


if __name__ == '__main__':
    unittest.main()
