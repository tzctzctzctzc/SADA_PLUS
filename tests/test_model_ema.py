import sys
import types
import unittest
import importlib.util
from pathlib import Path

import torch


sys.modules.setdefault('wandb', types.SimpleNamespace())
ROOT = Path(__file__).parents[1]
libs_module = types.ModuleType('libs')
libs_module.__path__ = [str(ROOT / 'libs')]
utils_module = types.ModuleType('libs.utils')
utils_module.__path__ = [str(ROOT / 'libs' / 'utils')]
postprocessing_module = types.ModuleType('libs.utils.postprocessing')
postprocessing_module.postprocess_results = lambda *args, **kwargs: None
sys.modules.setdefault('libs', libs_module)
sys.modules.setdefault('libs.utils', utils_module)
sys.modules.setdefault('libs.utils.postprocessing', postprocessing_module)
SPEC = importlib.util.spec_from_file_location(
    'libs.utils.train_utils', ROOT / 'libs' / 'utils' / 'train_utils.py'
)
train_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_utils)
ModelEma = train_utils.ModelEma
ModelEma2 = train_utils.ModelEma2


class SharedParameterModule(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.primary = torch.nn.Parameter(torch.tensor([value]))
        self.source_alias = self.primary
        self.target_alias = self.primary


class ModelEmaTest(unittest.TestCase):
    def _assert_shared_weight_updated_once(self, ema_cls):
        student = SharedParameterModule(1.0)
        ema = ema_cls(SharedParameterModule(0.0), decay=0.9)
        ema.update(student)
        self.assertAlmostEqual(ema.module.primary.item(), 0.1, places=6)

    def test_model_ema_updates_shared_parameter_once(self):
        self._assert_shared_weight_updated_once(ModelEma)

    def test_model_ema2_updates_shared_parameter_once(self):
        self._assert_shared_weight_updated_once(ModelEma2)


if __name__ == '__main__':
    unittest.main()
