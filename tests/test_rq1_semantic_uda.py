import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from tools.rq1_semantic_uda import (Recipe, Reverse, SemanticModel, CyclingSampler,
    conditional_loss, class_mean, fit)
from tools.run_rq1_semantic_uda import partitions, training_inputs, ROLES, run_item, sha, load_store


class SemanticUDATest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(6)
        self.recipe = Recipe(hidden=8, classes=3, batch_size=4, epochs=3,
            adv_start=1, warmup=1, val_every=1, patience=0, disc_dropout=0.)

    def test_gradient_reversal(self):
        x = torch.tensor([1., 2.], requires_grad=True)
        (Reverse.apply(x, .5) ** 2).sum().backward()
        torch.testing.assert_close(x.grad, -x.detach())

    def test_class_mean_equal_class_weight(self):
        v = torch.tensor([1., 1., 7.])
        loss, counts = class_mean(v, torch.tensor([0, 0, 1]), torch.ones(3, dtype=torch.bool), 3)
        self.assertEqual(loss.item(), 4.)
        self.assertEqual(counts.tolist(), [2., 1., 0.])

    def test_alignment_reaches_adapter_not_pseudo_classifier(self):
        model = SemanticModel(5, self.recipe)
        hs, _ = model(torch.randn(6, 5))
        ht, logits = model(torch.randn(6, 5))
        logits = logits * 0 + torch.tensor([6., 0., 0.])
        ht.retain_grad()
        loss, stat = conditional_loss(model, hs, torch.tensor([0, 1, 2, 0, 1, 2]), ht, logits)
        loss.backward()
        self.assertGreater(ht.grad.norm().item(), 0.)
        self.assertGreater(model.adapter[0].weight.grad.norm().item(), 0.)
        self.assertGreater(model.discriminator[0].weight.grad.norm().item(), 0.)
        self.assertIsNone(model.classifier.weight.grad)
        self.assertEqual(stat['target_accepted'].item(), 6)

    def test_uncertain_target_is_not_background(self):
        model = SemanticModel(5, self.recipe)
        hs, _ = model(torch.randn(6, 5))
        ht, _ = model(torch.randn(6, 5))
        ht.retain_grad()
        loss, stat = conditional_loss(model, hs, torch.tensor([0, 1, 2, 0, 1, 2]), ht, torch.zeros(6, 3))
        loss.backward()
        self.assertEqual(stat['target_accepted'].item(), 0)
        self.assertEqual(ht.grad.norm().item(), 0.)
        self.assertTrue(torch.isfinite(loss))

    def test_sampling_independent_of_training_rng(self):
        a, b = CyclingSampler(7, 4, 107), CyclingSampler(7, 4, 107)
        for _ in range(10):
            x = a.next()
            torch.rand(300)
            torch.testing.assert_close(x, b.next())

    def test_partitions_no_video_leak(self):
        rows = [dict(video_id=str(i), subset=role) for i, role in enumerate(ROLES)]
        cfg = {key: [role] for role, key in ROLES.items()}
        self.assertEqual(len(partitions(rows, cfg)), 4)
        rows[1]['video_id'] = rows[0]['video_id']
        with self.assertRaisesRegex(ValueError, 'leakage'):
            partitions(rows, cfg)

    def test_target_labels_absent_and_corruption_invariant(self):
        rows = [dict(video_id=str(i), subset=role, gt_label_id=0) for i, role in enumerate(ROLES)]
        cfg = {key: [role] for role, key in ROLES.items()}
        split = partitions(rows, cfg)
        features = np.ones((4, 5), dtype=np.float32)
        original = training_inputs(features, rows, split, 'cpu')
        del rows[1]['gt_label_id']
        del rows[3]['gt_label_id']
        hidden = training_inputs(features, rows, split, 'cpu')
        for a, b in zip(original, hidden):
            torch.testing.assert_close(a, b)

    def test_so_uda_same_initial_state_and_zero_alignment_trajectory(self):
        inputs = (torch.randn(7, 5), torch.arange(7) % 3,
                  torch.randn(11, 5), torch.randn(6, 5), torch.arange(6) % 3)
        r = replace(self.recipe, adv_weight=0.)
        so, hso, tso = fit(*inputs, r, 'so')
        uda, huda, tuda = fit(*inputs, r, 'uda')
        for name in ('adapter', 'classifier'):
            for a, b in zip(getattr(so, name).parameters(), getattr(uda, name).parameters()):
                torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-7)
        self.assertEqual(tso['updates'], tuda['updates'])
        self.assertEqual(tso['steps_per_epoch'], 3)
        self.assertEqual([x['source_val_loss'] for x in hso], [x['source_val_loss'] for x in huda])

    def test_patience_counts_epochs_and_checkpoints(self):
        inputs = (torch.randn(7, 5), torch.arange(7) % 3,
                  torch.randn(11, 5), torch.randn(6, 5), torch.arange(6) % 3)
        # 无学习时首轮即best，随后两个epoch无改善应第3轮终止。
        r = replace(self.recipe, epochs=10, patience=2, lr=1e-4, min_lr=1e-4, ema_decay=0.)
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d, patch('torch.optim.AdamW.step', return_value=None):
            _, h, term = fit(*inputs, r, 'so', Path(d))
            self.assertEqual(term['epoch'], 3)
            self.assertEqual(term['best_epoch'], 1)
            self.assertEqual(term['reason'], 'source_val_patience')
            self.assertTrue((Path(d) / 'best.pt').exists())
            self.assertTrue((Path(d) / 'latest.pt').exists())

    def test_end_to_end_artifacts_and_source_only_selection(self):
        import json
        rows = [dict(video_id=str(i), subset=role, gt_label_id=i % 3, candidate_id=str(i))
                for i, role in enumerate(list(ROLES) * 3)]
        split = partitions(rows, {key: [role] for role, key in ROLES.items()})
        features = np.random.default_rng(6).normal(size=(12, 5)).astype(np.float32)
        # 目标训练标签不存在；完整训练仍应成功，目标验证标签只写最终预测。
        for i in split['target_train']:
            del rows[i]['gt_label_id']
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'run'
            run_item(features, rows, split, self.recipe, 'uda', out, {}, 'cpu')
            record = json.loads((out / 'complete.json').read_text())
            for name, digest in record['sha256'].items():
                self.assertEqual(sha(out / name), digest)
            checkpoint = torch.load(out / 'best.pt', weights_only=False)
            self.assertIn('source_val_loss', checkpoint)
            self.assertNotIn('target_val_loss', checkpoint)
            with self.assertRaises(FileExistsError):
                run_item(features, rows, split, self.recipe, 'uda', out, {}, 'cpu')

    def test_incomplete_cache_rejected(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            prefix = Path(d) / 'cache'
            np.save(str(prefix) + '.features.npy', np.ones((3, 5), np.float16))
            np.save(str(prefix) + '.status.npy', np.array([1, 0, 1], np.uint8))
            Path(str(prefix) + '.meta.json').write_text(json.dumps({
                'schema': 'rq1-frozen-segment-features-v1', 'manifest_sha256': 'abc',
                'rows': 3, 'dimensions': 5, 'frame_count': 8}))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                load_store(prefix, 'abc', 3)


if __name__ == '__main__':
    unittest.main()
