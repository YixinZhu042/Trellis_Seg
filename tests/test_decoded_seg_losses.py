import importlib.util
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location('decoded_losses', Path(__file__).resolve().parents[1] / 'trainers/decoded_seg_losses.py')
losses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(losses)


class DecodedLossTests(unittest.TestCase):
    def test_clean_latent_formula_including_sigma(self):
        torch.manual_seed(2)
        clean, noise = torch.randn(8, 3), torch.randn(8, 3)
        for t in [0., .3, 1.]:
            sigma = .01
            noisy = (1-t)*clean+(sigma+(1-sigma)*t)*noise
            velocity = (1-sigma)*noise-clean
            torch.testing.assert_close(losses.recover_clean_latent(noisy, velocity, t, sigma), clean)

    def test_reprojection_matches_expanded_pixel_ce_and_gradient(self):
        logits = torch.tensor([[.1, 1., -.2], [.3, -.5, .8], [.1, .2, .3]], requires_grad=True)
        labels = torch.tensor([1, 2, 0])
        count = torch.tensor([2., 4., 7.])
        compact = losses.projected_pixel_ce(logits, labels, count)
        expanded = F.cross_entropy(logits[torch.tensor([0,0,1,1,1,1])], torch.tensor([1,1,2,2,2,2]))
        torch.testing.assert_close(compact, expanded)
        torch.testing.assert_close(torch.autograd.grad(compact, logits, retain_graph=True)[0],
                                   torch.autograd.grad(expanded, logits)[0])

    def test_boundary_loss_rewards_true_boundary_not_smoothing(self):
        labels = torch.tensor([1, 1, 2])
        adjacency = torch.tensor([[0, 1], [1, 2]])
        correct = F.one_hot(labels, 3).float()*10
        merged = F.one_hot(torch.ones(3, dtype=torch.long), 3).float()*10
        self.assertLess(losses.surface_boundary_loss(correct, labels, adjacency),
                        losses.surface_boundary_loss(merged, labels, adjacency))

    def test_all_three_losses_backpropagate_through_frozen_decoder(self):
        torch.manual_seed(3)
        decoder = torch.nn.Linear(4, 3).requires_grad_(False)
        latent = torch.randn(4, 4, requires_grad=True)
        logits = decoder(latent)
        sample = {'voxel_labels':torch.tensor([1,1,2,2]),'face_labels':torch.tensor([1,2,2]),
                  'face_area':torch.tensor([.4,.3,.3]), 'face_pixels':torch.tensor([2.,4.,0.]),
                  'face_knn':torch.tensor([[0,1],[1,2],[2,3]]),
                  'face_knn_weights':torch.full((3,2),.5), 'adjacency':torch.tensor([[0,1],[1,2]])}
        terms, _ = losses.decoded_task_losses(logits, sample)
        for term in terms.values():
            grad = torch.autograd.grad(term, latent, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.norm(), 0)
        sum(terms.values()).backward()
        self.assertTrue(all(p.grad is None for p in decoder.parameters()))

    def test_empty_supervision_is_finite_and_differentiable(self):
        x = torch.randn(3, 4, requires_grad=True)
        labels = torch.zeros(3, dtype=torch.long)
        loss = losses.balanced_voxel_ce(x, labels) + losses.projected_pixel_ce(x, labels, torch.zeros(3))
        loss += losses.surface_boundary_loss(x, labels, torch.empty(0, 2, dtype=torch.long))
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == '__main__':
    unittest.main()
