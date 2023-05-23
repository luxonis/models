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
                "name":"TestModel",
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
                        "name":"TestModel",
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
                        "name":"TestModel",
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
                "name":"TestModel",
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
                "name":"TestModel",
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
                "name":"TestModel",
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
                "name":"TestModel",
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
        self.assertEqual(cfg.get("trainer.num_sanity_val_steps"), 2) # one level deep
        self.assertEqual(cfg.get("dataset.local_path"), DATASET_PATH) # one level deep
        self.assertEqual(cfg.get("dataset.train_view"), "train") # get string
        self.assertEqual(cfg.get("train.batch_size"), 32) # get int
        self.assertEqual(cfg.get("train.skip_last_batch"), True) # get boolean
        self.assertEqual(cfg.get("train.preprocessing.train_image_size"), [256,256]) # get int array
        self.assertEqual(cfg.get("train.preprocessing.augmentations.0"), {"name": "Normalize", "params":{}}) # get dict

    def test_incorrect_get_key(self):
        """ Test using incorrect key in config get() method"""
        user_cfg_dict = {
            "model":{
                "name":"TestModel",
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
        keys = [
            "trainer.key1", # key doesn't exist in dict  (one level deep)
            "train.preprocessing.augmentations.0.key2", # key doesn't exist in dict 
            "train.preprocessing.augmentations.key2.0", # list index is string
            "train.optimizers.optimizer.name.key3", # key doesn't exist in dict (too deep)
            "exporter.openvino.scale_values.0.key4", # key doesn't exist in dict (too deep)
            "logger.logged_hyperparams.2" # list index out of bounds
            "train.preprocessing.augmentations.-1.name" # list index negative (middle)
            "logger.logged_hyperparams.-1" # list index negative (end)
        ]
        keys_should_fail = ["key1", "key2", "key2", "key3", "key4", "2", "-1", "-1"]
        for i, key in enumerate(keys):
            with self.subTest(i=i):
                with self.assertRaises(KeyError) as cm:
                    value = cfg.get(key)
                    self.assertIn(f"at level '{keys_should_fail[i]}'", str(cm)) 

    def test_no_nclasses(self):
        """ Test if no n_classes/params are defined in user config"""
        user_cfg_dict = {
            "model":{
                "name":"TestModel",
                "type": None,
                "pretrained": None,
                "backbone":{
                    "name": "MicroNet",
                    "pretrained": None,
                },
                "heads":[
                    {
                        "name": "ClassificationHead",
                        "loss": {
                            "name": "CrossEntropyLoss",
                        }
                    }
                ]
            },
            "dataset":{
                "local_path": DATASET_PATH
            },
        }
        cfg = Config(user_cfg_dict)

    def test_no_loss(self):
        """ Test if no loss is defined for a head or additional_head"""
        user_cfg_dict = {
            "model":{
                "name":"TestModel",
                "type": None,
                "pretrained": None,
                "backbone":{
                    "name": "MicroNet",
                    "pretrained": None,
                },
                "heads":[
                    {
                        "name": "ClassificationHead",
                    }
                ]
            },
            "dataset":{
                "local_path": DATASET_PATH
            },
        }
        with self.subTest(i=0):
            with self.assertRaises(KeyError):
                cfg = Config(user_cfg_dict)
        
        reset_env()
        user_cfg_dict2 = {
            "model":{
                "name":"TestModel",
                "type": "yolov6-n",
                "pretrained": None,
                "additional_heads":[
                    {
                        "name": "ClassificationHead",
                    }
                ]
            },
            "dataset":{
                "local_path": DATASET_PATH
            },
        }
        with self.subTest(i=1):
            with self.assertRaises(KeyError):
                cfg = Config(user_cfg_dict2)


if __name__ == "__main__":
    unittest.main()