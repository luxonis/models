import unittest
import torch
import luxonis_train.models.backbones as backbones
from luxonis_train.models.backbones import *


# update when new backbone is added
DEFAULT_INIT_VALUES = {
    "ContextSpatial": {},
    "EfficientNet": {},
    "EfficientRep": {},
    "MicroNet": {},
    "MobileNetV2": {},
    "MobileOne": {},
    "RepVGG": {},
    "ResNet18": {},
    "ReXNetV1_lite": {},
}


class BackboneTestCases(unittest.TestCase):
    def test_backbone_inference(self):
        """Tests inference on all backbones. Output of forward should be List[torch.Tensor]"""
        all_backbones = backbones.__all__
        input_shapes = [[1, 3, 256, 256], [1, 3, 512, 256]]
        for backbone_name in all_backbones:
            for input_shape in input_shapes:
                with self.subTest(backbone_name=backbone_name, input_shape=input_shape):
                    self.assertIn(backbone_name, DEFAULT_INIT_VALUES)

                    model = eval(backbone_name)(**DEFAULT_INIT_VALUES[backbone_name])
                    model.eval()
                    input = torch.zeros(input_shape)
                    outs = model(input)

                    self.assertIsInstance(outs, list)
                    for out in outs:
                        self.assertIsInstance(out, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
