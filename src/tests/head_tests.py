import unittest
import torch
import itertools
import luxonis_train.models.backbones as backbones
from luxonis_train.models.backbones import *
from luxonis_train.models.heads import *
from luxonis_train.utils.general import dummy_input_run

# update when new backbone or head is added
DEFAULT_BACKBONE_INIT_VALUES = {
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

DEFAULT_HEAD_INIT_VALUES = {
    "ClassificationHead": {},
    "MultiLabelClassificationHead": {},
    "SegmentationHead": {},
    "BiSeNetHead": {},
    "YoloV6Head": {},
    "IKeypoint": {},
}


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
                    self.assertIn(backbone_name, DEFAULT_BACKBONE_INIT_VALUES)
                    self.assertIn(head_name, DEFAULT_HEAD_INIT_VALUES)

                    backbone = eval(backbone_name)(
                        **DEFAULT_BACKBONE_INIT_VALUES[backbone_name]
                    )
                    input = torch.zeros(input_shape)
                    input_channels_shapes = dummy_input_run(backbone, input_shape)
                    backbone.eval()

                    head_params = {
                        "n_classes": 1,
                        "input_channels_shapes": input_channels_shapes,
                        "original_in_shape": input_shape,
                    } | DEFAULT_HEAD_INIT_VALUES[head_name]
                    head = eval(head_name)(**head_params)
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

        num_heads_attach_indices = [(2, [-1, -2, -3]), (3, [-1, -2]), (4, [-1])]

        for input_shape in input_shapes:
            for num_heads, neck_attach_indices in num_heads_attach_indices:
                for neck_attach_index in neck_attach_indices:
                    with self.subTest(
                        input_shape=input_shape,
                        num_heads=num_heads,
                        neck_attach_index=neck_attach_index,
                    ):
                        input = torch.zeros(input_shape)
                        input_channels_shapes = dummy_input_run(backbone, input_shape)
                        backbone.eval()
                        neck = RepPANNeck(
                            input_channels_shapes=input_channels_shapes,
                            num_heads=num_heads,
                            attach_index=neck_attach_index,
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
                        )

                        outs = backbone(input)
                        outs = neck(outs)
                        outs = head(outs)

                        self.assertIsInstance(outs, list)
                        self.assertEqual(len(outs), 3)

                        self.assertIsInstance(outs[0], list)
                        self.assertEqual(len(outs[0]), num_heads)
                        for out in outs[0]:
                            self.assertIsInstance(out, torch.Tensor)
                        self.assertIsInstance(outs[1], torch.Tensor)
                        self.assertIsInstance(outs[2], torch.Tensor)

    def test_yolov6_head_attach_index(self):
        """Tests attach_index parameter of YoloV6 head"""
        from luxonis_train.models.backbones import EfficientRep
        from luxonis_train.models.necks import RepPANNeck

        input_shape = [1, 3, 256, 256]
        backbone = EfficientRep()

        correct_values = [
            {
                "neck_num_heads": 3,
                "neck_attach_index": -1,
                "head_num_heads": 2,
                "head_attach_index": 1,
            },
            {
                "neck_num_heads": 2,
                "neck_attach_index": -2,
                "head_num_heads": 2,
                "head_attach_index": 0,
            },
            {
                "neck_num_heads": 4,
                "neck_attach_index": -1,
                "head_num_heads": 2,
                "head_attach_index": 2,
            },
        ]

        for curr_value in correct_values:
            with self.subTest(values=curr_value):
                input = torch.zeros(input_shape)
                input_channels_shapes = dummy_input_run(backbone, input_shape)
                backbone.eval()

                neck = RepPANNeck(
                    input_channels_shapes=input_channels_shapes,
                    num_heads=curr_value["neck_num_heads"],
                    attach_index=curr_value["neck_attach_index"],
                )
                input_channels_shapes = dummy_input_run(
                    neck, input_channels_shapes, multi_input=True
                )
                neck.eval()
                head = YoloV6Head(
                    n_classes=10,
                    input_channels_shapes=input_channels_shapes,
                    original_in_shape=input_shape,
                    num_heads=curr_value["head_num_heads"],
                    attach_index=curr_value["head_attach_index"],
                )

        wrong_values = [
            {
                "neck_num_heads": 3,
                "neck_attach_index": -1,
                "head_num_heads": 3,
                "head_attach_index": 1,
            },
            {
                "neck_num_heads": 2,
                "neck_attach_index": -2,
                "head_num_heads": 2,
                "head_attach_index": 1,
            },
            {
                "neck_num_heads": 4,
                "neck_attach_index": -1,
                "head_num_heads": 3,
                "head_attach_index": -2,
            },
        ]
        for curr_value in wrong_values:
            with self.subTest(values=curr_value):
                with self.assertRaises(ValueError):
                    input = torch.zeros(input_shape)
                    input_channels_shapes = dummy_input_run(backbone, input_shape)
                    backbone.eval()

                    neck = RepPANNeck(
                        input_channels_shapes=input_channels_shapes,
                        num_heads=curr_value["neck_num_heads"],
                        attach_index=curr_value["neck_attach_index"],
                    )
                    input_channels_shapes = dummy_input_run(
                        neck, input_channels_shapes, multi_input=True
                    )
                    neck.eval()
                    head = YoloV6Head(
                        n_classes=10,
                        input_channels_shapes=input_channels_shapes,
                        original_in_shape=input_shape,
                        num_heads=curr_value["head_num_heads"],
                        attach_index=curr_value["head_attach_index"],
                    )

    def test_ikeypoint_head(self):
        """Tests IKeypoint head together with EfficienRep backbone and RepPANNeck"""
        from luxonis_train.models.backbones import EfficientRep
        from luxonis_train.models.necks import RepPANNeck

        dummy_anchors = [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]

        input_shapes = [[1, 3, 256, 256], [1, 3, 512, 256]]
        backbone = EfficientRep()

        num_heads_attach_indices = [(2, [-1, -2, -3]), (3, [-1, -2]), (4, [-1])]

        for input_shape in input_shapes:
            for num_heads, neck_attach_indices in num_heads_attach_indices:
                for neck_attach_index in neck_attach_indices:
                    with self.subTest(
                        input_shape=input_shape,
                        num_heads=num_heads,
                        neck_attach_index=neck_attach_index,
                    ):
                        input = torch.zeros(input_shape)
                        input_channels_shapes = dummy_input_run(backbone, input_shape)
                        backbone.eval()
                        neck = RepPANNeck(
                            input_channels_shapes=input_channels_shapes,
                            num_heads=num_heads,
                            attach_index=neck_attach_index,
                        )
                        input_channels_shapes = dummy_input_run(
                            neck, input_channels_shapes, multi_input=True
                        )
                        neck.eval()

                        head = IKeypointHead(
                            n_classes=10,
                            n_keypoints=10,
                            input_channels_shapes=input_channels_shapes,
                            original_in_shape=input_shape,
                            num_heads=num_heads,
                            anchors=dummy_anchors[:num_heads],
                        )

                        outs = backbone(input)
                        outs = neck(outs)
                        outs = head(outs)

                        self.assertIsInstance(outs, list)
                        self.assertEqual(len(outs), 2)

                        self.assertIsInstance(outs[0], torch.Tensor)
                        self.assertIsInstance(outs[1], list)
                        self.assertEqual(len(outs[1]), num_heads)
                        for out in outs[1]:
                            self.assertIsInstance(out, torch.Tensor)

    def test_ikeypoint_head_params(self):
        """Tests parameters of IKeypoint head"""
        from luxonis_train.models.backbones import EfficientRep
        from luxonis_train.models.necks import RepPANNeck

        dummy_anchors = [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]

        input_shape = [1, 3, 256, 256]
        backbone = EfficientRep()

        correct_values = [
            {
                "neck_num_heads": 3,
                "neck_attach_index": -1,
                "head_num_heads": 2,
                "head_attach_index": 1,
                "head_anchor_len": 2,
            },
            {
                "neck_num_heads": 2,
                "neck_attach_index": -2,
                "head_num_heads": 2,
                "head_attach_index": 0,
                "head_anchor_len": 2,
            },
            {
                "neck_num_heads": 4,
                "neck_attach_index": -1,
                "head_num_heads": 2,
                "head_attach_index": 2,
                "head_anchor_len": 2,
            },
        ]

        for curr_value in correct_values:
            with self.subTest(values=curr_value):
                input = torch.zeros(input_shape)
                input_channels_shapes = dummy_input_run(backbone, input_shape)
                backbone.eval()

                neck = RepPANNeck(
                    input_channels_shapes=input_channels_shapes,
                    num_heads=curr_value["neck_num_heads"],
                    attach_index=curr_value["neck_attach_index"],
                )
                input_channels_shapes = dummy_input_run(
                    neck, input_channels_shapes, multi_input=True
                )
                neck.eval()
                head = IKeypointHead(
                    n_classes=10,
                    n_keypoints=10,
                    input_channels_shapes=input_channels_shapes,
                    original_in_shape=input_shape,
                    num_heads=curr_value["head_num_heads"],
                    attach_index=curr_value["head_attach_index"],
                    anchors=dummy_anchors[: curr_value["head_anchor_len"]],
                )

        wrong_values = [
            {
                "neck_num_heads": 3,
                "neck_attach_index": -1,
                "head_num_heads": 3,
                "head_attach_index": 1,
                "head_anchor_len": 2,
            },
            {
                "neck_num_heads": 2,
                "neck_attach_index": -2,
                "head_num_heads": 2,
                "head_attach_index": 1,
                "head_anchor_len": 3,
            },
            {
                "neck_num_heads": 4,
                "neck_attach_index": -1,
                "head_num_heads": 3,
                "head_attach_index": -2,
                "head_anchor_len": 4,
            },
        ]
        for curr_value in wrong_values:
            with self.subTest(values=curr_value):
                with self.assertRaises(ValueError):
                    input = torch.zeros(input_shape)
                    input_channels_shapes = dummy_input_run(backbone, input_shape)
                    backbone.eval()

                    neck = RepPANNeck(
                        input_channels_shapes=input_channels_shapes,
                        num_heads=curr_value["neck_num_heads"],
                        attach_index=curr_value["neck_attach_index"],
                    )
                    input_channels_shapes = dummy_input_run(
                        neck, input_channels_shapes, multi_input=True
                    )
                    neck.eval()
                    head = IKeypointHead(
                        n_classes=10,
                        n_keypoints=10,
                        input_channels_shapes=input_channels_shapes,
                        original_in_shape=input_shape,
                        num_heads=curr_value["head_num_heads"],
                        attach_index=curr_value["head_attach_index"],
                        anchors=dummy_anchors[: curr_value["head_anchor_len"]],
                    )

    def test_head_stride(self):
        """Tests stride of heads based on neck output for IKeypoint and YoloV6 head"""
        from luxonis_train.models.backbones import EfficientRep
        from luxonis_train.models.necks import RepPANNeck

        dummy_anchors = [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]

        input_shape = [1, 3, 256, 256]
        backbone = EfficientRep()

        values = [
            {
                "neck_num_heads": 4,
                "neck_attach_index": -1,
                "head_num_heads": 4,
                "head_attach_index": 0,
                "head_anchor_len": 4,
                "expected_stride": [4, 8, 16, 32],
            },
            {
                "neck_num_heads": 2,
                "neck_attach_index": -1,
                "head_num_heads": 2,
                "head_attach_index": 0,
                "head_anchor_len": 2,
                "expected_stride": [4, 8],
            },
            {
                "neck_num_heads": 3,
                "neck_attach_index": -2,
                "head_num_heads": 2,
                "head_attach_index": 1,
                "head_anchor_len": 2,
                "expected_stride": [8, 16],
            },
        ]

        for curr_value in values:
            with self.subTest(values=curr_value):
                input = torch.zeros(input_shape)
                input_channels_shapes = dummy_input_run(backbone, input_shape)
                backbone.eval()

                neck = RepPANNeck(
                    input_channels_shapes=input_channels_shapes,
                    num_heads=curr_value["neck_num_heads"],
                    attach_index=curr_value["neck_attach_index"],
                )
                input_channels_shapes = dummy_input_run(
                    neck, input_channels_shapes, multi_input=True
                )
                neck.eval()
                head = IKeypointHead(
                    n_classes=10,
                    n_keypoints=10,
                    input_channels_shapes=input_channels_shapes,
                    original_in_shape=input_shape,
                    num_heads=curr_value["head_num_heads"],
                    attach_index=curr_value["head_attach_index"],
                    anchors=dummy_anchors[: curr_value["head_anchor_len"]],
                )
                self.assertEqual(head.stride.tolist(), curr_value["expected_stride"])

                head = YoloV6Head(
                    n_classes=10,
                    input_channels_shapes=input_channels_shapes,
                    original_in_shape=input_shape,
                    num_heads=curr_value["head_num_heads"],
                    attach_index=curr_value["head_attach_index"],
                )
                self.assertEqual(head.stride.tolist(), curr_value["expected_stride"])


if __name__ == "__main__":
    unittest.main()
