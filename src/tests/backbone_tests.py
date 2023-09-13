import unittest
import torch

from luxonis_train.utils.registry import BACKBONES

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
        input_shapes = [[1, 3, 256, 256], [1, 3, 512, 256]]
        for backbone_name in BACKBONES.module_dict:
            for input_shape in input_shapes:
                with self.subTest(backbone_name=backbone_name, input_shape=input_shape):
                    init_params = DEFAULT_INIT_VALUES.get(backbone_name, {})

                    model = BACKBONES.get(backbone_name)(**init_params)
                    model.eval()
                    input = torch.zeros(input_shape)
                    outs = model(input)

                    self.assertIsInstance(outs, list)
                    for out in outs:
                        self.assertIsInstance(out, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
