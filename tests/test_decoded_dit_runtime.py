import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from utils.decoded_dit_runtime import (FeatureOutputModel, distributed_topology,
                                       training_sample_index)
from scripts.train_dit_decoded import (compose_training_loss, normalize_aligned_features,
                                       save_training_checkpoint)

ROOT = Path(__file__).resolve().parents[1]


class TinySparseDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Linear(2, 2, bias=False)
        self.mv = torch.nn.Linear(2, 2, bias=False)

    def forward(self, x, drop=False):
        from torch.utils.checkpoint import checkpoint
        def block(tensor):
            value = self.main(tensor)
            if not drop:
                value = value + self.mv(tensor)
            return value
        value = checkpoint(block, x, use_reentrant=False)
        return SimpleNamespace(feats=value)


def _ddp_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + str(Path(directory) / 'rendezvous'),
                            rank=rank, world_size=2)
    try:
        torch.manual_seed(10)
        model = TinySparseDenoiser()
        reference = TinySparseDenoiser()
        reference.load_state_dict(model.state_dict())
        decoder = torch.nn.Linear(2, 2, bias=False).requires_grad_(False)
        ddp = DistributedDataParallel(FeatureOutputModel(model), find_unused_parameters=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=.1, momentum=.9)
        for step in range(4):
            optimizer.zero_grad(set_to_none=True)
            expected_optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                # Some ranks drop conditioning on the synchronized microbatch;
                # others use it, exercising DDP's unused-parameter traversal.
                drop = step == 3 or (rank + micro + step) % 3 == 0
                x = torch.tensor([[rank + 1., micro + 1.]])
                from contextlib import nullcontext
                with ddp.no_sync() if micro == 0 else nullcontext():
                    pred = ddp(x, drop=drop)
                    loss = pred.square().mean()
                    if rank == 0:
                        aux = decoder(pred).square().mean()
                        grad = torch.autograd.grad(aux, pred, retain_graph=True)[0]
                        assert torch.isfinite(grad).all()
                        loss = loss + .2 * aux
                    (loss / 2).backward()
            for other_rank in range(2):
                for micro in range(2):
                    drop = step == 3 or (other_rank + micro + step) % 3 == 0
                    x = torch.tensor([[other_rank + 1., micro + 1.]])
                    pred = reference(x, drop=drop).feats
                    loss = pred.square().mean()
                    if other_rank == 0:
                        loss += .2 * decoder(pred).square().mean()
                    (loss / 4).backward()
            optimizer.step()
            expected_optimizer.step()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
            assert all(p.grad is None for p in decoder.parameters())
            if step == 1:
                save_training_checkpoint(Path(directory), 2, model, optimizer, {'mode': 'decoded'},
                                         {'rank': rank}, rank, 2)
                resume = torch.load(Path(directory) / 'trainer_latest.pt', weights_only=False)
                # Force restoration to matter before continuing to compare
                # against the uninterrupted reference optimizer trajectory.
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(10)
                optimizer.state.clear()
                model.load_state_dict(resume['model'])
                optimizer.load_state_dict(resume['optimizer'])
        metadata = {'world_size': 2, 'mode': 'decoded'}
        save_training_checkpoint(Path(directory), 4, model, optimizer, metadata,
                                 {'rank': rank}, rank, 2)
        checkpoint = torch.load(Path(directory) / 'trainer_latest.pt', weights_only=False)
        assert checkpoint['rank_rng'] == [{'rank': 0}, {'rank': 1}]
        assert checkpoint['step'] == 4 and checkpoint['metadata'] == metadata
        for key, value in model.state_dict().items():
            torch.testing.assert_close(checkpoint['model'][key], value)
        assert not any(Path(directory).glob('*.tmp'))
        (Path(directory) / f'pass_rank{rank}').write_text('ok')
    finally:
        dist.destroy_process_group()


class DecodedDistributedTests(unittest.TestCase):
    def test_loss_composition_keeps_teacher_and_auxiliary_terms(self):
        flow = torch.tensor(2., requires_grad=True)
        teacher = torch.tensor(3., requires_grad=True)
        auxiliary = torch.tensor(5., requires_grad=True)
        loss = compose_training_loss(
            flow, teacher_loss=teacher, teacher_weight=.1,
            auxiliary_loss=auxiliary, auxiliary_gate=.2)
        self.assertAlmostEqual(float(loss.detach()), 3.3, places=6)
        loss.backward()
        self.assertAlmostEqual(float(flow.grad), 1.)
        self.assertAlmostEqual(float(teacher.grad), .1)
        self.assertAlmostEqual(float(auxiliary.grad), .2)

    def test_single_rank_checkpoint_keeps_ema_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters())
            ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
            save_training_checkpoint(Path(directory), 1, model, optimizer, {'mode': 'decoded'},
                                     {'rank': 0}, 0, 1, ema_state=ema)
            checkpoint = torch.load(Path(directory) / 'trainer_latest.pt', weights_only=False)
            self.assertIsNotNone(checkpoint['ema'])
            for name, value in ema.items():
                torch.testing.assert_close(checkpoint['ema'][name], value)
            self.assertTrue((Path(directory) / 'dit_ema_step000001.pt').is_file())

    def test_unmatched_shape_condition_stays_zero_after_normalization(self):
        features = torch.tensor([[3., 5.], [0., 0.]])
        normalized = normalize_aligned_features(
            features, torch.tensor([1., 1.]), torch.tensor([2., 4.]),
            torch.tensor([True, False]))
        torch.testing.assert_close(normalized[0], torch.tensor([1., 1.]))
        torch.testing.assert_close(normalized[1], torch.zeros(2))

    def test_topology_and_global_data_sharding(self):
        self.assertEqual(distributed_topology({'WORLD_SIZE': '16', 'RANK': '9', 'LOCAL_RANK': '1'}), (9, 1, 16))
        for world in [1, 2, 8, 16]:
            for step in [1, 2, 5]:
                indices = [training_sample_index(step, micro, rank, world, 2, 320, seed=0)
                           for micro in range(2) for rank in range(world)]
                # Every global microbatch has disjoint ranks, and a full epoch
                # is a bijection even though each epoch is deterministically reshuffled.
                self.assertEqual(len(indices), len(set(indices)))
        epoch = [training_sample_index(1 + i // 16, 0, i % 16, 16, 1, 320, seed=7)
                 for i in range(320)]
        self.assertEqual(set(epoch), set(range(320)))
        with self.assertRaises(ValueError):
            distributed_topology({'WORLD_SIZE': '2', 'RANK': '2'})

    def test_taiji_dry_run_and_invalid_node_rank(self):
        env = dict(os.environ, SEG_DIT_DRY_RUN='1', SEG_DIT_MODE='decoded',
                   SEG_DIT_PROJECT_DIR=str(ROOT), SEG_DIT_GPUS_PER_NODE='8', SEG_DIT_ACCUMULATE='1')
        cmd = ['bash', str(ROOT / 'scripts/train_dit_decoded_taiji.sh'), '2', '1', '10.0.0.1']
        result = subprocess.run(cmd, env=env, text=True, capture_output=True, check=True)
        self.assertIn('global_batch=16', result.stdout)
        self.assertIn('--node_rank 1', result.stdout)
        self.assertIn('--nproc_per_node 8', result.stdout)
        self.assertIn('--mode decoded', result.stdout)
        env['SEG_DIT_MODE'] = 'flow'
        flow = subprocess.run(cmd, env=env, text=True, capture_output=True, check=True)
        self.assertIn('--mode flow', flow.stdout)
        self.assertIn('--master_port 12556', flow.stdout)
        result = subprocess.run(cmd[:-2] + ['2', '10.0.0.1'], env=env, capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_two_cpu_ranks_match_global_batch_and_save_once(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_ddp_worker, args=(directory,), nprocs=2, join=True)
            self.assertTrue((Path(directory) / 'pass_rank0').exists())
            self.assertTrue((Path(directory) / 'pass_rank1').exists())
            metadata = json.loads((Path(directory) / 'checkpoint_latest.json').read_text())
            self.assertEqual(metadata['step'], 4)


if __name__ == '__main__':
    unittest.main()
