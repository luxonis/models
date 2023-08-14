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

    def test_yolov6_head(self):
        """Tests YoloV6 head together with EfficienRep backbone and RepPANNeck"""
        from luxonis_train.models.backbones import EfficientRep
        from luxonis_train.models.necks import RepPANNeck

        input_shapes = [[1, 3, 256, 256], [1, 3, 512, 256]]
        backbone = EfficientRep()

        num_heads_offset_pairs = [(2, [0, 1, 2]), (3, [0, 1]), (4, [0])]

        for input_shape in input_shapes:
            for num_heads, offsets in num_heads_offset_pairs:
                for offset in offsets:
                    input = torch.zeros(input_shape)
                    input_channels_shapes = dummy_input_run(backbone, input_shape)
                    backbone.eval()
                    neck = RepPANNeck(
                        input_channels_shapes=input_channels_shapes, num_heads=num_heads
                    )
                    input_channels_shapes = dummy_input_run(
                        neck, input_channels_shapes, multi_input=True
                    )
                    neck.eval()

                    head = YoloV6Head(
                        n_classes=10,
                        input_channels_shapes=input_channels_shapes,
                        original_in_shape=input_shape,
                        num_heads=num_heads,
                        offset=offset,
                    )

                    outs = backbone(input)
                    outs = neck(outs)
                    outs = head(outs)

                    self.assertIsInstance(outs, list)
                    self.assertEqual(len(outs), 3)

                    self.assertIsInstance(outs[0], list)
                    for out in outs[0]:
                        self.assertIsInstance(out, torch.Tensor)
                    self.assertIsInstance(outs[1], torch.Tensor)
                    self.assertIsInstance(outs[2], torch.Tensor)


if __name__ == "__main__":
    unittest.main()
