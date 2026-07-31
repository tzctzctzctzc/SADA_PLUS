import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch


SADA_DIR = Path(__file__).parents[1] / "libs/models/SADA"
PACKAGE = "_begtr_sada_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(SADA_DIR)]
sys.modules.setdefault(PACKAGE, package)
for module_name in ("weight_init", "blocks", "aux_boundary_head"):
    full_name = "{}.{}".format(PACKAGE, module_name)
    spec = importlib.util.spec_from_file_location(
        full_name, SADA_DIR / "{}.py".format(module_name)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)

AuxBoundaryHead = sys.modules[
    "{}.aux_boundary_head".format(PACKAGE)
].AuxBoundaryHead


class AuxBoundaryHeadTest(unittest.TestCase):
    def test_outputs_class_agnostic_maps_per_level(self):
        head = AuxBoundaryHead(4, 8, num_layers=2)
        features = [torch.randn(2, 4, 7), torch.randn(2, 4, 4)]
        masks = [
            torch.ones(2, 1, 7, dtype=torch.bool),
            torch.ones(2, 1, 4, dtype=torch.bool),
        ]
        starts, ends = head(features, masks)
        self.assertEqual([tuple(item.shape) for item in starts], [(2, 7), (2, 4)])
        self.assertEqual([tuple(item.shape) for item in ends], [(2, 7), (2, 4)])

    def test_padding_is_masked(self):
        head = AuxBoundaryHead(4, 8)
        features = [torch.randn(1, 4, 5)]
        masks = [torch.tensor([[[True, True, True, False, False]]])]
        starts, ends = head(features, masks)
        self.assertTrue(torch.equal(starts[0][0, 3:], torch.zeros(2)))
        self.assertTrue(torch.equal(ends[0][0, 3:], torch.zeros(2)))


if __name__ == "__main__":
    unittest.main()
