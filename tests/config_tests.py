import unittest
import os
from luxonis_train.utils.config import Config

DATASET_PATH = "/home/klemen/luxonis/test_datasets/datasets/coco-2017-person"

def reset_env():
    """ Removes Config() instance from current envirnoment. """
    Config.clear_instance()

class ConfigFileTestCases(unittest.TestCase):
    def tearDown(self):
        reset_env()
    
    def test_path_init(self):
        """ Test passing path to test_config.yaml"""
        relative_cfg_pth = "test_config.yaml"
        user_cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), relative_cfg_pth))
        try:
            cfg = Config(user_cfg_path)
        except Exception as e:
            self.fail(f"Config() raised exception: {e}!")
    
    def test_path_doesnt_exists(self):
        """ Test passing path that doesn't exists """
        with self.assertRaises(ValueError):
            cfg = Config("incorrect_path.yaml")

    def test_incorrect_yaml_cfg(self):
        """ Test passing path to yaml that doens't include model definition """
        relative_cfg_pth = "test_config_fail.yaml"
        user_cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), relative_cfg_pth))
        with self.assertRaises(KeyError):
            cfg = Config(user_cfg_path)

class ConfigDictTestCases(unittest.TestCase):
    def tearDown(self):
        reset_env()

    def test_dir_init(self):
        """ Test passing dir to Config"""
        user_cfg_dict = {
            "model":{
                "name":"SimpleDetection",
                "type": None,
                "pretrained": None,
                "backbone":{
                    "name": "MicroNet",
                    "pretrained": None
                },
                "heads": [
                    {
                        "name": "ClassificationHead",
                        "params":{
                            "n_classes": None
                        },
                        "loss": {
                            "name": "CrossEntropyLoss",
                            "params": None
                        }
                    }
                ]
            },
            "dataset":{
                "local_path": DATASET_PATH
            }
        }
        try:
            cfg = Config(user_cfg_dict)
        except Exception as e:
            self.fail(f"Config() raised exception: {e}!")
    
    def test_empty_config_init(self):
        """ Test passing no parameters to config init"""
        with self.assertRaises(ValueError):
            cfg = Config()

    def test_incorrect_dir(self):
        """ Test passing dir without model definition """
        user_cfg_dict = {
            "dataset":{
                "local_path": DATASET_PATH
            }
        }
        with self.assertRaises(KeyError):
            cfg = Config(user_cfg_dict)
    
class ConfigValuesTestCases(unittest.TestCase):    
    def tearDown(self):
        reset_env()

    def test_predefined_yolo(self):
        """ Test constructing config from predefined model"""
        yolo_versions = ["n", "s", "t"]
        for i, version in enumerate(yolo_versions):
            with self.subTest(i=i):
                reset_env()
                curr_type = f"yolov6-{version}"
                user_cfg_dict = {
                    "model":{
                        "name":"SimpleDetection",
                        "type": curr_type,
                        "pretrained": None,
                        "params":{
                            "n_classes": None
                        }
                    },
                    "dataset":{
                        "local_path": DATASET_PATH
                    }
                }
                try:
                    cfg = Config(user_cfg_dict)
                except Exception as e:
                    self.fail(f"Config() raised exception: {e}!")

    def test_incorrect_type(self):
        """ Test setting inccorect model type"""
        types = ["yolov6-l", "false-model"]
        for i, curr_type in enumerate(types):
            with self.subTest(i=i):
                reset_env()
                user_cfg_dict = {
                    "model":{
                        "name":"SimpleDetection",
                        "type": curr_type,
                        "pretrained": None,
                        "params":{
                            "n_classes": None
                        }
                    },
                    "dataset":{
                        "local_path": DATASET_PATH
                    }
                }
                with self.assertRaises(ValueError):
                    cfg = Config(user_cfg_dict)

    def test_incorrect_dataset(self):
        """ Test providing incorrect dataset path"""
        empty_dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "empty_dataset"))
        user_cfg_dict = {
            "model":{
                "name":"SimpleDetection",
                "type": "yolov6-n",
                "pretrained": None,
                "params":{
                    "n_classes": None
                }
            },
            "dataset":{
                "local_path": empty_dataset_path
            }
        }
        with self.assertRaises(ValueError):
            cfg = Config(user_cfg_dict)

    def test_incorrect_n_classes(self):
        """ Test setting incorrect n_classes"""
        user_cfg_dict = {
            "model":{
                "name":"SimpleDetection",
                "type": "yolov6-n",
                "pretrained": None,
                "params":{
                    "n_classes": 10
                }
            },
            "dataset":{
                "local_path": DATASET_PATH
            }
        }
        with self.assertRaises(KeyError):
            cfg = Config(user_cfg_dict)

    def test_new_keyvalue_pair(self):
        """ Test creating new key-value pair in config """
        user_cfg_dict = {
            "model":{
                    "name":"SimpleDetection",
                    "type": "yolov6-n",
                    "pretrained": None,
                    "params":{
                        "n_classes": None
                    }
                },
            "dataset":{
                "local_path": DATASET_PATH
            },
            "train":{
                "optimizers":{
                    "scheduler":{
                        "params":{
                            "new_key":"new_value"
                        }
                    }
                }
            }
        }
        with self.assertWarns(Warning):
            cfg = Config(user_cfg_dict)
            self.assertEqual(cfg.get("train.optimizers.scheduler.params.new_key"), "new_value")

    def test_get_value(self):
        """ Test Config get() method on different types and different depths"""
        user_cfg_dict = {
            "model":{
                    "name":"SimpleDetection",
                    "type": "yolov6-n",
                    "pretrained": None,
                    "params":{
                        "n_classes": None
                    }
                },
            "dataset":{
                "local_path": DATASET_PATH
            },
        }
        cfg = Config(user_cfg_dict)
        self.assertEqual(cfg.get("trainer.num_sanity_val_steps"), 2)
        self.assertEqual(cfg.get("dataset.local_path"), DATASET_PATH)
        self.assertEqual(cfg.get("dataset.train_view"), "train")
        self.assertEqual(cfg.get("train.batch_size"), 32)
        self.assertEqual(cfg.get("train.skip_last_batch"), True)
        self.assertEqual(cfg.get("train.preprocessing.train_image_size"), [256,256])

    def test_incorrect_get_key(self):
        """ Test using incorrect key in config get() method"""
        user_cfg_dict = {
            "model":{
                    "name":"SimpleDetection",
                    "type": "yolov6-n",
                    "pretrained": None,
                    "params":{
                        "n_classes": None
                    }
                },
            "dataset":{
                "local_path": DATASET_PATH
            },
        }
        cfg = Config(user_cfg_dict)
        keys = ["trainer.key1", "train.preprocessing.augmentations.0.key2",
            "train.preprocessing.augmentations.key2.0", "train.optimizers.optimizer.name.key3",
            "exporter.openvino.scale_values.0.key4", "logger.logged_hyperparams.2"]
        for i, key in enumerate(keys):
            with self.subTest(i=i):
                with self.assertRaises(KeyError):
                    value = cfg.get(key)
                    print(value)


if __name__ == "__main__":
    unittest.main()