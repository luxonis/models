import os
import yaml
import warnings
import json
import re
import sys
from typing import Union
from copy import deepcopy

from luxonis_ml.data import LuxonisDataset, BucketType, BucketStorage
from luxonis_train.utils.filesystem import LuxonisFileSystem
from luxonis_train.utils.registry import HEADS


class Config:
    """Singleton class which checks and merges user config with default one and provides access to its values"""

    _db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config_db"))

    def __new__(cls, cfg=None):
        if not hasattr(cls, "instance"):
            if cfg is None:
                raise ValueError("Provide either config path or config dictionary.")

            cls.instance = super(Config, cls).__new__(cls)
            cls.instance._load(cfg)

        return cls.instance

    def __repr__(self):
        return json.dumps(self._data, indent=4)

    @classmethod
    def clear_instance(cls):
        """Clears all singleton instances, should be only used for unit-testing"""
        if hasattr(cls, "instance"):
            del cls.instance

    def get_data(self):
        """Returns a deepcopy of current data dict"""
        return deepcopy(self._data)

    def save_data(self, path: str):
        """Saves data dict to yaml file"""
        with open(path, "w+") as f:
            yaml.dump(self._data, f, default_flow_style=False)

    def override_config(self, args: str):
        """Overrides config values with ones specifid by --override string.
        If last key is not matched to config, creates a new (key, value) pair.
        """
        if len(args) == 0:
            return

        args = remove_chars_inside_brackets(
            args
        )  # remove characters that are not needed
        items = args.split(" ")

        if len(items) % 2 != 0:
            raise ValueError(
                "Parameters passed by --override should be in 'key value' shape but one value is missing."
            )

        for i in range(0, len(items), 2):
            key_merged = items[i]
            value = items[i + 1]
            last_key, last_sub_dict = self._config_iterate(key_merged, only_warn=True)

            # should skip if _config_iterate returns None
            if last_key is None and last_sub_dict is None:
                continue
            # check if value represents something other than string
            value = parse_string_to_types(value)
            last_sub_dict[last_key] = value

    def get(self, key_merged: str):
        """Returns value from config based on the key"""
        last_key, last_sub_dict = self._config_iterate(key_merged, only_warn=False)
        return last_sub_dict[last_key]

    def validate_config_exporter(self):
        """Validates 'exporter' block in config"""
        if not self._data["exporter"]:
            raise ValueError("No 'exporter' section in config specified.")

        if not self._data["exporter"]["export_weights"]:
            warnings.warn(
                "No 'export_weights' speficied in config file, using random weights instead."
            )

    def validate_config_tuner(self):
        """Validates 'tuner' block in config"""
        if not self._data["tuner"]:
            raise ValueError("No 'tuner' section in config specified.")
        if not ("params" in self._data["tuner"] and self._data["tuner"]["params"]):
            raise ValueError(
                "Nothing to tune as no tuner params specified in the config."
            )
        # TODO: and more checks if needed

    def _load(self, cfg: Union[str, dict]):
        """Performs complete loading and validation of the config"""
        with open(os.path.join(self._db_path, "config_all.yaml"), "r") as f:
            base_cfg = yaml.load(f, Loader=yaml.SafeLoader)

        if isinstance(cfg, str):
            from dotenv import load_dotenv

            load_dotenv()  # load environment variables needed for authorization

            fs = LuxonisFileSystem(cfg)
            buffer = fs.read_to_byte_buffer()
            user_cfg = yaml.load(buffer, Loader=yaml.SafeLoader)
            if fs.is_mlflow:
                warnings.warn(
                    "Setting `project_id` and `run_id` to config's MLFlow run"
                )
                # set logger parameters to continue run
                if "logger" in user_cfg:
                    user_cfg["logger"]["project_id"] = fs.experiment_id
                    user_cfg["logger"]["run_id"] = fs.run_id
                else:
                    base_cfg["logger"]["project_id"] = fs.experiment_id
                    base_cfg["logger"]["run_id"] = fs.experiment_id

        elif isinstance(cfg, dict):
            user_cfg = cfg
        else:
            raise ValueError("Provided cfg is neither path(string) or dictionary.")

        self._user_cfg = user_cfg
        self._data = self._merge_configs(base_cfg, user_cfg)
        self._load_model_config(user_cfg)
        self._validate_dataset_classes()
        self._validate_config()
        print("Config loaded.")

    def _merge_configs(self, base_cfg: dict, user_cfg: dict):
        """Merges user config with base config"""
        if base_cfg is None:
            base_cfg = {}

        for key, value in user_cfg.items():
            # Load model config after merge
            if key == "model":
                continue
            if isinstance(value, dict):
                # If the value is a dictionary, recurse
                base_cfg[key] = self._merge_configs(base_cfg.get(key, {}), value)
            else:
                # Otherwise, overwrite the value in the base dictionary
                if key not in base_cfg:
                    warnings.warn(
                        f"New (key,value) pair added to config: ({key},{value})"
                    )
                base_cfg[key] = value

        return base_cfg

    def _load_model_config(self, cfg: dict):
        """Loads model config from user config"""
        model_cfg = cfg.get("model")
        if model_cfg is None:
            raise KeyError("Model config must be present in config file.")
        # check if we should load predefined model
        if model_cfg["type"] is not None:
            warnings.warn(
                "Loading predefined model overrides local config of backbone, neck and head."
            )
            model_cfg = self._load_predefined_model(model_cfg)
        else:
            if model_cfg.get("additional_heads"):
                warnings.warn(
                    "Additional heads won't be taken into account if you don't specify model type."
                    + "Move them to 'heads' if you want to use them."
                )
                model_cfg.pop("additional_heads")

        self._data["model"] = model_cfg

    def _load_predefined_model(self, model_cfg: dict):
        """Loads predefined model config from db"""
        model_type = model_cfg["type"].lower()
        if model_type.startswith("yolov6"):
            return self._load_yolov6_cfg(model_cfg)
        else:
            raise ValueError(f"{model_type} not supported")

    def _load_yolov6_cfg(self, model_cfg: dict):
        """Loads predefined YoloV6 config from db"""
        predefined_cfg_path = model_cfg["type"].lower() + ".yaml"
        full_path = os.path.join(self._db_path, predefined_cfg_path)
        if not os.path.isfile(full_path):
            raise ValueError(
                f"There is no predefined config for this {model_cfg['type']} type."
            )

        with open(full_path, "r") as f:
            predefined_cfg = yaml.load(f, Loader=yaml.SafeLoader)

        model_cfg["backbone"] = predefined_cfg["backbone"]
        model_cfg["neck"] = predefined_cfg["neck"]
        model_cfg["heads"] = predefined_cfg["heads"]

        model_params = (
            model_cfg["params"]
            if "params" in model_cfg and model_cfg["params"] is not None
            else {}
        )
        for key, value in model_params.items():
            if value is None:
                continue
            if key == "n_classes":
                model_cfg["heads"][0]["params"]["n_classes"] = value
                model_cfg["heads"][0]["loss"]["params"]["n_classes"] = value
            # refactored in further PRs
            if key == "num_heads":
                model_cfg["neck"]["params"]["num_heads"] = value
                model_cfg["heads"][0]["params"]["num_heads"] = value

        if "additional_heads" in model_cfg and isinstance(
            model_cfg["additional_heads"], list
        ):
            model_cfg["heads"].extend(model_cfg["additional_heads"])
            model_cfg.pop("additional_heads")

        return model_cfg

    def _config_iterate(self, key_merged: str, only_warn: bool = False):
        """Iterates over config based on key_merged and returns last key and
        sub dict if it mathces structure
        """
        sub_keys = key_merged.split(".")
        sub_dict = self._data
        success = True
        key = sub_keys[0]  # needed if search only one level deep
        for key in sub_keys[:-1]:
            if isinstance(sub_dict, list):  # check if key should be list index
                if key.isdigit():
                    key = int(key)
                    # list index out of bounds
                    if key >= len(sub_dict) or key < 0:
                        success = False
                else:
                    success = False
            elif key not in sub_dict:
                # key not present in sub_dict
                success = False

            if not success:
                if only_warn:
                    warnings.warn(
                        f"Key '{key_merged}' not matched to config (at level '{key}'). Skipping."
                    )
                    return None, None
                else:
                    raise KeyError(
                        f"Key '{key_merged}' not matched to config (at level '{key}')"
                    )

            sub_dict = sub_dict[key]
        # return last_key in correct format (int or string)
        success = True
        if isinstance(sub_dict, dict):
            last_key = sub_keys[-1]
            if last_key not in sub_dict:
                key = last_key
                success = False
        elif isinstance(sub_dict, list):
            if sub_keys[-1].isdigit():
                last_key = int(sub_keys[-1])
                # list index out of bounds
                if last_key >= len(sub_dict) or last_key < 0:
                    success = False
            else:
                key = sub_keys[-1]
                success = False
        else:
            key = sub_keys[-1]
            success = False

        if not success and not only_warn:
            raise KeyError(
                f"Key '{key_merged}' not matched to config (at level '{key}')"
            )

        return last_key, sub_dict

    def _validate_dataset_classes(self):
        """Validates config to used datasets, overrides n_classes if needed"""

        with LuxonisDataset(
            team_id=self._data["dataset"]["team_id"],
            dataset_id=self._data["dataset"]["dataset_id"],
            bucket_type=eval(self._data["dataset"]["bucket_type"]),
            bucket_storage=eval(self._data["dataset"]["bucket_storage"]),
        ) as dataset:
            classes, classes_by_task = dataset.get_classes()

            if not classes:
                raise ValueError("Provided dataset doesn't have any classes.")

            model_cfg = self._data["model"]
            for head in model_cfg["heads"]:
                if not ("params" in head and head["params"]):
                    head["params"] = {}

                curr_n_classes = head["params"].get("n_classes", None)
                label_type = get_head_label_types(head["name"])[0]
                dataset_n_classes = len(classes_by_task[label_type.value])
                if curr_n_classes is None:
                    warnings.warn(
                        f"Inheriting 'n_classes' parameter from dataset. Setting it to {dataset_n_classes}"
                    )
                elif curr_n_classes != dataset_n_classes:
                    raise KeyError(
                        f"Number of classes in config ({curr_n_classes}) doesn't match number of "
                        + f"classes in dataset ({dataset_n_classes})"
                    )
                head["params"]["n_classes"] = dataset_n_classes

                # also set n_classes to loss params
                if not ("loss" in head and head["loss"]):
                    # loss definition should be present in every head
                    raise KeyError("Loss must be defined for every head.")
                if not ("params" in head["loss"] and head["loss"]["params"]):
                    head["loss"]["params"] = {}
                head["loss"]["params"]["n_classes"] = dataset_n_classes

    def _validate_config(self):
        """Validates whole config based on specified rules"""
        model_cfg = self._data["model"]
        model_predefined = model_cfg["type"] != None
        backbone_specified = "backbone" in model_cfg and model_cfg["backbone"]
        neck_specified = "neck" in model_cfg and model_cfg["neck"]
        heads_specified = "heads" in model_cfg and isinstance(model_cfg["heads"], list)

        if not model_predefined:
            if not (backbone_specified and heads_specified):
                raise KeyError("Backbone and at least 1 head must be specified.")

            if "params" in model_cfg and model_cfg["params"]:
                warnings.warn(
                    "Model-wise parameters won't be taken into account if you don't specify model type."
                )

        if model_cfg["pretrained"] and model_cfg["backbone"]["pretrained"]:
            warnings.warn(
                "Weights of the backbone will be overridden by whole model weights."
            )

        n_heads = len(model_cfg["heads"])

        # handle main_head_index
        if not (0 <= self._data["train"]["main_head_index"] < n_heads):
            raise KeyError(
                "Specified index of main head ('main_head_index') is out of range."
            )

        # handle freeze_modules and losses sections
        if "train" in self._user_cfg and self._user_cfg["train"]:
            # if freeze_modules used in user cfg than 'heads' must match number of heads
            if (
                "freeze_modules" in self._user_cfg["train"]
                and self._user_cfg["train"]["freeze_modules"]
            ):
                if len(self._user_cfg["train"]["freeze_modules"]["heads"]) != n_heads:
                    raise KeyError(
                        "Number of heads in the model doesn't match number of heads in 'freeze_modules.heads'."
                    )
            # handle loss weights (similar as freeze_modules)
            if (
                "losses" in self._user_cfg["train"]
                and self._user_cfg["train"]["losses"]
            ):
                if len(self._user_cfg["train"]["losses"]["weights"]) != n_heads:
                    raise KeyError(
                        "Number of losses in the model doesn't match number of losses in 'losses.weights'."
                    )

        # check if heads in freeze_modules and losses matches number of heads of the model
        if len(self._data["train"]["freeze_modules"]["heads"]) != n_heads:
            self._data["train"]["freeze_modules"]["heads"] = [False] * n_heads

        if len(self._data["train"]["losses"]["weights"]) != n_heads:
            self._data["train"]["losses"]["weights"] = [1] * n_heads

        # handle normalize augmentation - append to other augmentations
        if not (
            "augmentations" in self._data["train"]["preprocessing"]
            and self._data["train"]["preprocessing"]["augmentations"]
        ):
            self._data["train"]["preprocessing"]["augmentations"] = []
        if self._data["train"]["preprocessing"]["normalize"]["active"]:
            normalize_present = any(
                filter(
                    lambda x: x["name"] == "Normalize",
                    self._data["train"]["preprocessing"]["augmentations"],
                )
            )
            if not normalize_present:
                self._data["train"]["preprocessing"]["augmentations"].append(
                    {
                        "name": "Normalize",
                        "params": self._data["train"]["preprocessing"]["normalize"][
                            "params"
                        ]
                        if "params" in self._data["train"]["preprocessing"]["normalize"]
                        and self._data["train"]["preprocessing"]["normalize"]["params"]
                        else {},
                    }
                )

        # handle optimizer and scheduler params - set to empty dict if None
        if not self._data["train"]["optimizers"]["optimizer"]["params"]:
            self._data["train"]["optimizers"]["optimizer"]["params"] = {}
        if not self._data["train"]["optimizers"]["scheduler"]["params"]:
            self._data["train"]["optimizers"]["scheduler"]["params"] = {}

        # handle setting num_workers to 0 for Mac and Windows
        if sys.platform == "win32" or sys.platform == "darwin":
            self._data["train"]["num_workers"] = 0

        # handle IKeypointHead with anchors=None by generating them from dataset
        ikeypoint_head_indices = [
            i
            for i, head in enumerate(self._data["model"]["heads"])
            if head["name"] == "IKeypointHead"
        ]
        if len(ikeypoint_head_indices):
            from torch.utils.data import DataLoader
            from luxonis_ml.loader import ValAugmentations, LuxonisLoader
            from luxonis_train.utils.boxutils import anchors_from_dataset

            for i in ikeypoint_head_indices:
                head = self._data["model"]["heads"][i]
                anchors = head["params"].get("anchors", -1)
                if anchors is None:
                    with LuxonisDataset(
                        team_id=self._data["dataset"]["team_id"],
                        dataset_id=self._data["dataset"]["dataset_id"],
                        bucket_type=eval(self._data["dataset"]["bucket_type"]),
                        bucket_storage=eval(self._data["dataset"]["bucket_storage"]),
                    ) as dataset:
                        val_augmentations = ValAugmentations(
                            image_size=self._data["train"]["preprocessing"][
                                "train_image_size"
                            ],
                            augmentations=[{"name": "Normalize", "params": {}}],
                            train_rgb=self._data["train"]["preprocessing"]["train_rgb"],
                            keep_aspect_ratio=self._data["train"]["preprocessing"][
                                "keep_aspect_ratio"
                            ],
                        )
                        loader = LuxonisLoader(
                            dataset,
                            view=self._data["dataset"]["train_view"],
                            augmentations=val_augmentations,
                        )
                        pytorch_loader = DataLoader(
                            loader,
                            batch_size=self._data["train"]["batch_size"],
                            num_workers=self._data["train"]["num_workers"],
                            collate_fn=loader.collate_fn,
                        )
                        num_heads = head["params"].get("num_heads", 3)
                        proposed_anchors = anchors_from_dataset(
                            pytorch_loader, n_anchors=num_heads * 3
                        )
                        head["params"]["anchors"] = proposed_anchors.reshape(
                            -1, 6
                        ).tolist()


def get_head_label_types(head_str: str):
    """Returns all label types defined as head class attributes"""
    return HEADS.get(head_str).label_types


def remove_chars_inside_brackets(string):
    """Find and remove all spaces, single and double quotes inside substring which starts
    with [ and ends with ] character
    """
    # Find substrings enclosed in square brackets
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, string)
    # Remove spaces, single quotes, and double quotes within each substring
    for match in matches:
        updated_match = re.sub(r'[\s\'"]', "", match)
        string = string.replace(match, updated_match)
    return string


def parse_string_to_types(input_str: str):
    """Parse input strings to different data type if it matches some rule"""
    if input_str.lstrip("-").isdigit():  # check if is digit
        out = int(input_str)
    elif input_str.lstrip("-").replace(".", "", 1).isdigit():
        out = float(input_str)
    elif input_str.lower() in ["none", "null"]:  # check if None
        out = None
    elif input_str.lower() == "true":  # check for bool values
        out = True
    elif input_str.lower() == "false":
        out = False
    elif input_str.startswith("[") and input_str.endswith("]"):  # check if list
        sub_inputs = input_str[1:-1].split(",")
        out = []
        for sub_input in sub_inputs:
            out.append(parse_string_to_types(sub_input))  # parse every sub-item
    else:
        out = input_str  # if nothing matches then it's a string
    return out
