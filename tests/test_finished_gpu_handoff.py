import unittest,tempfile
from pathlib import Path
from tools.baseline_protocol import save,sha
from tools.finished_gpu_handoff import available_indices

class HandoffTests(unittest.TestCase):
    def test_no_release_keeps_two_cards(self):
        self.assertEqual(available_indices([0,1],None),[0,1])
    def test_requires_both_final_evaluations_and_no_active_task(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);audit=root/'audit.json';release=root/'release.json'
            save(audit,dict(status='passed',count=2));save(root/'state.json',dict(phase='train',active={'train':1}))
            save(release,dict(status='released-after-independent-final-audit',gpus=[2,3],root=str(root),audit=str(audit),audit_sha256=sha(audit)))
            with self.assertRaises(ValueError):available_indices([0,1],release)
            save(root/'state.json',dict(phase='complete',active={}))
            self.assertEqual(available_indices([0,1],release),[0,1,2,3])
            save(audit,dict(status='passed',count=1))
            with self.assertRaises(ValueError):available_indices([0,1],release)

if __name__=='__main__':unittest.main()
