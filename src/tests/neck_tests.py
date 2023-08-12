import unittest
import torch
from luxonis_train.models.necks import *
from luxonis_train.utils.general import dummy_input_run


class NeckTestCases(unittest.TestCase):
    def test_reppan_inference(self):
        """Tests inference on RepPAN neck with EfficienRep backbone. Output of forward should be List[torch.Tensor]"""
        from luxonis_train.models.backbones import EfficientRep

        input_shapes = [[1, 3, 256, 256], [1, 3, 512, 256]]
        backbone = EfficientRep()

        for input_shape in input_shapes:
            for num_heads in [2, 3, 4]:
                with self.subTest(input_shape=input_shape, num_heads=num_heads):
                    input = torch.zeros(input_shape)
                    input_channels_shapes = dummy_input_run(backbone, input_shape)
                    neck = RepPANNeck(
                        input_channels_shapes=input_channels_shapes, num_heads=num_heads
                    )

                    outs = backbone(input)
                    outs = neck(outs)

                    self.assertIsInstance(outs, list)
                    for out in outs:
                        self.assertIsInstance(out, torch.Tensor)

    def test_reppan_incorrect_num_heas(self):
        """Tests num_heads parameter of RepPAN neck"""
        input_channels_shapes = [
            [1, 32, 56, 56],
            [1, 64, 28, 28],
            [1, 128, 14, 14],
            [1, 256, 7, 7],
        ]
        for num_heads in [1, 5, -1]:
            with self.subTest(num_heads=num_heads):
                with self.assertRaises(ValueError):
                    neck = RepPANNeck(
                        input_channels_shapes=input_channels_shapes, num_heads=num_heads
                    )


if __name__ == "__main__":
    unittest.main()
