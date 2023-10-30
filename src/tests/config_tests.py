import unittest
import os
import json
from pydantic import ValidationError
from dotenv import load_dotenv

from luxonis_train.utils.config import ConfigHandler
from luxonis_train.utils.config.config import *


TEAM_ID = "luxonis_team_id"
DATASET_ID = "64ee246f8163b4b5778fc769"


def reset_env():
    """Removes ConfigHandler() instance from current envirnoment."""
    ConfigHandler.clear_instance()


class ConfigFileTestCases(unittest.TestCase):
    def tearDown(self):
        reset_env()

    def test_path_init(self):
        """Test passing path to test_config.yaml"""
        relative_cfg_pth = "test_config.yaml"
        user_cfg_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), relative_cfg_pth)
        )
        try:
            cfg = ConfigHandler(user_cfg_path)
        except Exception as e:
            self.fail(f"ConfigHandler() raised exception: {e}!")

    def test_path_doesnt_exists(self):
        """Test passing path that doesn't exists"""
        with self.assertRaises(FileNotFoundError):
            cfg = ConfigHandler("incorrect_path.yaml")

    def test_incorrect_yaml_cfg(self):
        """Test passing path to yaml that doens't include model definition"""
        relative_cfg_pth = "test_config_fail.yaml"
        user_cfg_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), relative_cfg_pth)
        )
        with self.assertRaises(ValidationError):
            cfg = ConfigHandler(user_cfg_path)


class ConfigDictTestCases(unittest.TestCase):
    def tearDown(self):
        reset_env()

    def test_dir_init(self):
        """Test passing dir to ConfigHandler"""
        user_cfg_dict = {
            "model": {
                "name": "TestModel",
                "backbone": {"name": "MicroNet", "pretrained": None},
                "heads": [
                    {
                        "name": "ClassificationHead",
                        "loss": {"name": "CrossEntropyLoss"},
                    }
                ],
            },
            "dataset": {
                "team_id": TEAM_ID,
                "dataset_id": DATASET_ID,
            },
        }
        try:
            cfg = ConfigHandler(user_cfg_dict)
        except Exception as e:
            self.fail(f"ConfigHandler() raised exception: {e}!")

    def test_empty_config_init(self):
        """Test passing no parameters to config init"""
        with self.assertRaises(ValueError):
            cfg = ConfigHandler()

    def test_incorrect_dir(self):
        """Test passing dir without model definition"""
        user_cfg_dict = {
            "dataset": {
                "team_id": TEAM_ID,
                "dataset_id": DATASET_ID,
            }
        }
        with self.assertRaises(ValidationError):
            cfg = ConfigHandler(user_cfg_dict)


class ConfigValuesTestCases(unittest.TestCase):
    def tearDown(self):
        reset_env()

    def test_predefined_yolo(self):
        """Test constructing config from predefined model"""
        yolo_versions = ["n", "t", "s"]
        for i, version in enumerate(yolo_versions):
            with self.subTest(i=i):
                reset_env()
                curr_predefined_model = f"yolov6-{version}"
                user_cfg_dict = {
                    "model": {
                        "name": "TestModel",
                        "predefined_model": curr_predefined_model,
                    },
                    "dataset": {
                        "team_id": TEAM_ID,
                        "dataset_id": DATASET_ID,
                    },
                }
                try:
                    cfg = ConfigHandler(user_cfg_dict)
                except Exception as e:
                    self.fail(f"ConfigHandler() raised exception: {e}!")

    def test_incorrect_predefined_model(self):
        """Test setting inccorect predefined model"""
        predefined_models = ["yolov6-l", "false-model"]
        for i, curr_predefined_model in enumerate(predefined_models):
            with self.subTest(i=i):
                reset_env()
                user_cfg_dict = {
                    "model": {
                        "name": "TestModel",
                        "predefined_model": curr_predefined_model,
                    },
                    "dataset": {
                        "team_id": TEAM_ID,
                        "dataset_id": DATASET_ID,
                    },
                }
                with self.assertRaises(ValueError):
                    cfg = ConfigHandler(user_cfg_dict)

    def test_incorrect_dataset(self):
        """Test providing incorrect dataset path"""
        empty_dataset_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "empty_dataset")
        )
        user_cfg_dict = {
            "model": {
                "name": "TestModel",
                "type": "yolov6-n",
                "pretrained": None,
                "params": {"n_classes": None},
            },
            "dataset": {"local_path": empty_dataset_path},
        }
        with self.assertRaises(ValueError):
            cfg = ConfigHandler(user_cfg_dict)

    def test_incorrect_n_classes(self):
        """Test setting incorrect n_classes"""
        user_cfg_dict = {
            "model": {
                "name": "TestModel",
                "predefined_model": "yolov6-n",
                "params": {"n_classes": 10},
            },
            "dataset": {
                "team_id": TEAM_ID,
                "dataset_id": DATASET_ID,
            },
        }
        with self.assertRaises(ValueError):
            cfg = ConfigHandler(user_cfg_dict)

    def test_new_keyvalue_pair(self):
        """Test creating new key-value pair in config"""
        user_cfg_dict = {
            "model": {
                "name": "TestModel",
                "predefined_model": "yolov6-n",
            },
            "dataset": {
                "team_id": TEAM_ID,
                "dataset_id": DATASET_ID,
            },
            "train": {
                "optimizers": {"scheduler": {"params": {"new_key": "new_value"}}}
            },
        }
        cfg = ConfigHandler(user_cfg_dict)
        self.assertEqual(
            cfg.get("train.optimizers.scheduler.params.new_key"), "new_value"
        )

    def test_get_value(self):
        """Test ConfigHandler get() method on different types and different depths"""
        user_cfg_dict = {
            "model": {
                "name": "TestModel",
                "predefined_model": "yolov6-n",
            },
            "dataset": {
                "team_id": TEAM_ID,
                "dataset_id": DATASET_ID,
            },
        }
        cfg = ConfigHandler(user_cfg_dict)
        self.assertEqual(cfg.get("trainer.num_sanity_val_steps"), 2)  # one level deep
        self.assertEqual(cfg.get("dataset.dataset_id"), DATASET_ID)  # one level deep
        self.assertEqual(cfg.get("dataset.train_view"), "train")  # get string
        self.assertEqual(cfg.get("train.batch_size"), 32)  # get int
        self.assertEqual(cfg.get("train.skip_last_batch"), True)  # get boolean
        self.assertEqual(
            cfg.get("train.preprocessing.train_image_size"), [256, 256]
        )  # get int array
        self.assertEqual(
            cfg.get("train.preprocessing.augmentations.0").model_dump(),
            {"name": "Normalize", "params": {}},
        )  # get config object (use model_dump() to compare to dict)

    def test_incorrect_get_key(self):
        """Test using incorrect key in config get() method"""
        user_cfg_dict = {
            "model": {
                "name": "TestModel",
                "predefined_model": "yolov6-n",
            },
            "dataset": {
                "team_id": TEAM_ID,
                "dataset_id": DATASET_ID,
            },
        }
        cfg = ConfigHandler(user_cfg_dict)
        keys = [
            "trainer.key1",  # key doesn't exist in dict  (one level deep)
            "train.preprocessing.augmentations.0.key2",  # key doesn't exist in dict
            "train.preprocessing.augmentations.key2.0",  # list index is string
            "train.optimizers.optimizer.name.key3",  # key doesn't exist in dict (too deep)
            "exporter.scale_values.0.key4",  # key doesn't exist in dict (too deep)
            "logger.logged_hyperparams.2",  # list index out of bounds
            "train.preprocessing.augmentations.-2.name",  # list index negative (middle)
            "logger.logged_hyperparams.-1",  # list index negative (end)
        ]

        for i, key in enumerate(keys):
            with self.subTest(i=i):
                with self.assertWarns(Warning):
                    value = cfg.get(key)

    def test_auto_nclasses(self):
        """Test if n_classes not specified and automatically set"""
        user_cfg_dict = {
            "model": {
                "name": "TestModel",
                "backbone": {
                    "name": "MicroNet",
                },
                "heads": [
                    {
                        "name": "ClassificationHead",
                        "loss": {
                            "name": "CrossEntropyLoss",
                        },
                    }
                ],
            },
            "dataset": {
                "team_id": TEAM_ID,
                "dataset_id": DATASET_ID,
            },
        }
        cfg = ConfigHandler(user_cfg_dict)
        self.assertNotEqual(cfg.get("model.heads.0.params.n_classes"), None)

    def test_no_loss(self):
        """Test if no loss is defined for a head or additional_head"""
        user_cfg_dict = {
            "model": {
                "name": "TestModel",
                "backbone": {
                    "name": "MicroNet",
                },
                "heads": [
                    {
                        "name": "ClassificationHead",
                    }
                ],
            },
            "dataset": {
                "team_id": TEAM_ID,
                "dataset_id": DATASET_ID,
            },
        }
        with self.subTest(i=0):
            with self.assertRaises(ValidationError):
                cfg = ConfigHandler(user_cfg_dict)

        reset_env()
        user_cfg_dict2 = {
            "model": {
                "name": "TestModel",
                "predefined_model": "yolov6-n",
                "additional_heads": [
                    {
                        "name": "ClassificationHead",
                    }
                ],
            },
            "dataset": {
                "team_id": TEAM_ID,
                "dataset_id": DATASET_ID,
            },
        }
        with self.subTest(i=1):
            with self.assertRaises(ValidationError):
                cfg = ConfigHandler(user_cfg_dict2)

    def test_override(self):
        """Test config override with a string"""
        user_cfg_dict = {
            "model": {
                "name": "TestModel",
                "predefined_model": "yolov6-n",
            },
            "dataset": {
                "team_id": TEAM_ID,
                "dataset_id": DATASET_ID,
            },
        }

        override_tests = {
            "trainer.accelerator cpu": [
                "trainer.accelerator",
                "auto",
                "cpu",
            ],  # one level deep
            "dataset.train_view test": [
                "dataset.train_view",
                "train",
                "test",
            ],  # string
            "train.batch_size 16": ["train.batch_size", 32, 16],  # int
            "train.preprocessing.train_rgb False": [
                "train.preprocessing.train_rgb",
                True,
                False,
            ],  # bool
            "logger.logged_hyperparams [train.test]": [
                "logger.logged_hyperparams",
                ["train.epochs", "train.batch_size"],
                ["train.test"],
            ],  # list of strings
            "logger.logged_hyperparams ['train.test']": [
                "logger.logged_hyperparams",
                ["train.epochs", "train.batch_size"],
                ["train.test"],
            ],  # not needed characters
            "train.preprocessing.train_image_size [512, 512]": [
                "train.preprocessing.train_image_size",
                [256, 256],
                [512, 512],
            ],  # list of ints
            "exporter.export_image_size.0 128": [
                "exporter.export_image_size.0",
                256,
                128,
            ],  # int inside list
        }

        for i, (override_str, (key, previous_val, new_val)) in enumerate(
            override_tests.items()
        ):
            reset_env()
            with self.subTest(i=i):
                cfg = ConfigHandler(user_cfg_dict)
                initial_cfg = cfg.get_data()
                cfg.override_config(override_str)
                current_val = cfg.get(key)
                self.assertEqual(current_val, new_val)
                cfg.override_config(f"{key} {previous_val}")
                self.assertEqual(cfg.get_data(), initial_cfg)

    def test_override(self):
        """Test config override with a string"""
        user_cfg_dict = {
            "model": {
                "name": "TestModel",
                "predefined_model": "yolov6-n",
            },
            "dataset": {
                "team_id": TEAM_ID,
                "dataset_id": DATASET_ID,
            },
        }

        override_tests = {
            '{"trainer.accelerator":"cpu"}': [
                "trainer.accelerator",
                "auto",
                "cpu",
            ],  # one level deep
            '{"dataset.train_view":"test"}': [
                "dataset.train_view",
                "train",
                "test",
            ],  # string
            '{"train.batch_size":16}': ["train.batch_size", 32, 16],  # int
            '{"train.preprocessing.train_rgb": false}': [
                "train.preprocessing.train_rgb",
                True,
                False,
            ],  # bool
            '{"logger.logged_hyperparams": ["train.test"]}': [
                "logger.logged_hyperparams",
                ["train.epochs", "train.batch_size"],
                ["train.test"],
            ],  # list of strings
            '{"train.preprocessing.train_image_size": [512,512]}': [
                "train.preprocessing.train_image_size",
                [256, 256],
                [512, 512],
            ],  # list of ints
            '{"exporter.export_image_size.0": 128}': [
                "exporter.export_image_size.0",
                256,
                128,
            ],  # int inside list
            '{"tuner.storage": {"active": false, "storage_type": "remote"}}': [
                "tuner.storage",
                StorageConfig(active=True, storage_type="local"),
                StorageConfig(active=False, storage_type="remote"),
            ],  # custom class
        }

        for i, (override_str, (key, previous_val, new_val)) in enumerate(
            override_tests.items()
        ):
            reset_env()
            with self.subTest(i=i):
                cfg = ConfigHandler(user_cfg_dict)
                initial_cfg = cfg.get_data()
                cfg.override_config(json.loads(override_str))
                current_val = cfg.get(key)
                self.assertEqual(current_val, new_val)
                cfg.override_config({key: previous_val})
                self.assertEqual(cfg.get_data(), initial_cfg)

    def test_override_add(self):
        """Test adding new items/keys through override"""
        user_cfg_dict = {
            "model": {
                "name": "TestModel",
                "predefined_model": "yolov6-n",
            },
            "dataset": {
                "team_id": TEAM_ID,
                "dataset_id": DATASET_ID,
            },
        }

        override_tests = {
            '{"logger.logged_hyperparams.2" : "new_hyperparam"}': [
                "logger.logged_hyperparams.2",
                "new_hyperparam",
            ],  # add string to list
            '{"model.heads.1":{"name":"Test","loss":{"name":"TestLoss"}}}': [
                "model.heads.1",
                ModelHeadConfig(name="Test", loss=LossModuleConfig(name="TestLoss")),
            ],  # add object to list
            '{"train.optimizers.optimizer.params.new_param" : "new_value"}': [
                "train.optimizers.optimizer.params.new_param",
                "new_value",
            ],  # add key-value to dict
        }

        for i, (override_str, (key, new_val)) in enumerate(override_tests.items()):
            reset_env()
            with self.subTest(i=i):
                cfg = ConfigHandler(user_cfg_dict)
                cfg.override_config(json.loads(override_str))
                current_val = cfg.get(key)
                self.assertEqual(current_val, new_val)


if __name__ == "__main__":
    load_dotenv()
    unittest.main()
