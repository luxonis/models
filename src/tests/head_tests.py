import unittest
import torch
import itertools
import luxonis_train.models.backbones as backbones
from luxonis_train.models.backbones import *
from luxonis_train.models.heads import *
from luxonis_train.utils.general import dummy_input_run


class HeadTestCases(unittest.TestCase):
    def test_simple_heads(self):
        """Tests combination of all backbones with all simple heads. Output should be torch.Tensor"""
        all_backbones = backbones.__all__
        # NOTE: if new head created add it to the list
        simple_heads = [
            "ClassificationHead",
            "MultiLabelClassificationHead",
            "SegmentationHead",
            "BiSeNetHead",
        ]
        combinations = list(itertools.product(all_backbones, simple_heads))

        input_shapes = [[1, 3, 256, 256], [1, 3, 512, 256]]
        for backbone_name, head_name in combinations:
            for input_shape in input_shapes:
                with self.subTest(
                    backbone_name=backbone_name,
                    head_name=head_name,
                    input_shape=input_shape,
                ):
                    backbone = eval(backbone_name)(**{})
                    input = torch.zeros(input_shape)
                    input_channels_shapes = dummy_input_run(backbone, input_shape)
                    backbone.eval()

                    head = eval(head_name)(
                        **{
                            "n_classes": 1,
                            "input_channels_shapes": input_channels_shapes,
                            "original_in_shape": input_shape,
                        }
                    )
                    head.eval()

                    outs = backbone(input)
                    outs = head(outs)

                    self.assertIsInstance(outs, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
