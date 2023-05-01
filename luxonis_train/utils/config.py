import os
import yaml
import warnings
import json
import torch
from luxonis_ml import LuxonisDataset

class Config:
    _db_path = "configs/db"

    def __new__(cls, cfg=None):
        if not hasattr(cls, 'instance'):
            if cfg is None:
                raise ValueError("Provide either config path or config dictionary.")
            
            cls.instance = super(Config, cls).__new__(cls)
            cls.instance._load(cfg)

        return cls.instance

    def __repr__(self):
        return json.dumps(self._data, indent=4)

    def override_config(self, args):
        if len(args) == 0:
            return 
        items = args.split(" ")
        if len(items) % 2 != 0:
            raise ValueError("Parameters passed by --override should be in 'key value' shape but one value is missing.")

        for i in range(0, len(items), 2):
            key_merged = items[i]
            value = items[i+1]
            (iter_success, key), last_key, last_sub_dict = self._config_iterate(key_merged)
            last_matched = (isinstance(last_sub_dict, dict) and last_key in last_sub_dict) or \
                (isinstance(last_sub_dict, list) and 0<=last_key<len(last_sub_dict))
            if not(iter_success and last_matched):
                warnings.warn(f"Key '{key_merged}' not matched to config "+
                    f"(at level '{key if not iter_success else last_key}'). Skipping.")
                continue
            last_sub_dict[last_key] = int(value) if value.isdigit() else value

    def get(self, key_merged):
        (iter_success, key), last_key, last_sub_dict = self._config_iterate(key_merged)
        last_matched = (isinstance(last_sub_dict, dict) and last_key in last_sub_dict) or \
            (isinstance(last_sub_dict, list) and 0<=last_key<len(last_sub_dict))
        
        if not(iter_success and last_matched):
            raise KeyError(f"Key '{key_merged}' not matched to config "+
                f"(at level '{key if not iter_success else last_key}')")
        return last_sub_dict[last_key]

    def validate_config_exporter(self):
        if not self._data["exporter"]:
            raise KeyError("No 'exporter' section in config specified.")

        if not self._data["exporter"]["export_weights"]:
            raise KeyError("No 'export_weights' speficied in config file.")
        
    def _config_iterate(self, key_merged):
        sub_keys = key_merged.split(".")
        sub_dict = self._data
        success = True
        key = sub_keys[0] # needed if only search only one level deep
        for key in sub_keys[:-1]:
            if isinstance(sub_dict, list): # check if key should be list index 
                key = int(key)
                if key >= len(sub_dict) or key < 0:
                    success = False
                    break
            elif key not in sub_dict:
                success = False
                break
            sub_dict = sub_dict[key]

        last_key = sub_keys[-1] if isinstance(sub_dict, dict) else int(sub_keys[-1])
        return (success, key), last_key, sub_dict


    def _load(self, cfg):
        with open(os.path.join(self._db_path, "config_all.yaml"), "r") as f:
            base_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
        if isinstance(cfg, str):
            with open(cfg, "r") as f:
                user_cfg = yaml.load(f, Loader=yaml.SafeLoader)
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

    def _merge_configs(self, base_cfg, user_cfg):
        for key, value in user_cfg.items():
            # Load model config after merge
            if key == "model":
                continue
            if isinstance(value, dict):
                # If the value is a dictionary, recurse
                base_cfg[key] = self._merge_configs(base_cfg.get(key, {}), value)
            else:
                # Otherwise, overwrite the value in the base dictionary
                if key in base_cfg:
                    base_cfg[key] = value
                else:
                    warnings.warn(f"Key '{key}' not matched to config. Skipping.")
        return base_cfg

    def _load_model_config(self, cfg):
        model_cfg = cfg.get("model")
        if model_cfg is None:
            raise KeyError("Model config must be present in config file.")
        # check if we should load predefined model
        if model_cfg["type"] is not None:
            warnings.warn("Loading predefined model overrides local config of backbone, neck and head.")
            model_cfg = self._load_predefined_model(model_cfg)
        else:
            if model_cfg.get("additional_heads"):
                warnings.warn("Additional heads won't be taken into account if you don't specify model type."+
                    "Move them to 'heads' if you want to use them.")
                model_cfg.pop("additional_heads")

        self._data["model"] = model_cfg

    def _load_predefined_model(self, model_cfg):
        model_type = model_cfg["type"].lower()
        if model_type.startswith("yolov6"):
            return self._load_yolov6_cfg(model_cfg)
        else:
            raise RuntimeError(f"{model_type} not supported")

    def _load_yolov6_cfg(self, model_cfg):
        predefined_cfg_path = model_cfg["type"].lower() +".yaml"
        with open(os.path.join(self._db_path, predefined_cfg_path), "r") as f:
            predefined_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        
        model_cfg["backbone"] = predefined_cfg["backbone"]
        model_cfg["neck"] = predefined_cfg["neck"]
        model_cfg["heads"] = predefined_cfg["heads"]

        model_params = model_cfg["params"] if "params" in model_cfg and \
            model_cfg["params"] is not None else {}
        for key, value in model_params.items():
            if value is None:
                continue
            if key == "n_classes":
                model_cfg["heads"][0]["params"]["n_classes"] = value
                model_cfg["heads"][0]["loss"]["params"]["n_classes"] = value

        if "additional_heads" in model_cfg and isinstance(model_cfg["additional_heads"], list):
            model_cfg["heads"].extend(model_cfg["additional_heads"])
            model_cfg.pop("additional_heads")
        
        return model_cfg

    def _validate_dataset_classes(self):
        with LuxonisDataset(
            local_path=self._data["dataset"]["local_path"],
            s3_path=self._data["dataset"]["s3_path"]
        ) as dataset:
            dataset_n_classes = len(dataset.classes)
            # TODO: implement per task number of classes
            # for key in dataset.classes_by_task:
            #     print(key, len(dataset.classes_by_task[key]))

            model_cfg = self._data["model"]
            for head in model_cfg["heads"]:
                if not ("params" in head and head["params"]):
                    head["params"] = {}

                curr_n_classes = head["params"].get("n_classes", None)
                if curr_n_classes is None:
                    warnings.warn(f"Inheriting 'n_classes' parameter from dataset. Setting it to {dataset_n_classes}")
                elif curr_n_classes != dataset_n_classes:
                    warnings.warn(f"Number of classes in config ({curr_n_classes}) doesn't match number of"+
                        f"classes in dataset ({dataset_n_classes}). Setting it to {dataset_n_classes}")
                head["params"]["n_classes"] = dataset_n_classes

                # also set n_classes to loss params
                if not("paras" in head["loss"] and head["loss"]["params"]):
                    head["loss"]["params"] = {}
                head["loss"]["params"]["n_classes"] = dataset_n_classes

    def _validate_config(self):
        model_cfg = self._data["model"]
        model_predefined = model_cfg["type"] != None
        backbone_specified = "backbone" in model_cfg and model_cfg["backbone"]
        neck_specified = "neck" in model_cfg and model_cfg["neck"]
        heads_specified = "heads" in model_cfg and isinstance(model_cfg["heads"], list)

        if not model_predefined:
            if not (backbone_specified and heads_specified):
                raise KeyError("Backbone and at least 1 head must be specified.")

            if "params" in model_cfg and model_cfg["params"]:
                warnings.warn("Model-wise parameters won't be taken into account if you don't specify model type.")

        for head in model_cfg["heads"]:
            if not("n_classes" in head["params"] and head["params"]["n_classes"]):
                raise KeyError("Head 'n_classes' param must be defined for every head.")
            if not ("loss" in head and head["loss"]):
                raise KeyError("Loss must be defined for every head.")

        if model_cfg["pretrained"] and model_cfg["backbone"]["pretrained"]:
            warnings.warn("Weights of the backbone will be overridden by whole model weights.")

        n_heads = len(model_cfg["heads"])
        # handle freeze_modules and losses sections
        if "train" in self._user_cfg and self._user_cfg["train"]:
            # if freeze_modules used in user cfg than 'heads' must match number of heads
            if "freeze_modules" in self._user_cfg["train"] and self._user_cfg["train"]["freeze_modules"]:
                if len(self._user_cfg["train"]["freeze_modules"]["heads"]) != n_heads:
                    raise KeyError("Number of heads in the model doesn't match number of heads in 'freeze_modules.heads'.")
            # handle loss weights (similar as freeze_modules)
            if "losses" in self._user_cfg["train"] and self._user_cfg["train"]["losses"]:
                if len(self._user_cfg["train"]["losses"]["weights"]) != n_heads:
                    raise KeyError("Number of losses in the model doesn't match number of losses in 'losses.weights'.")
        
        # check if heads in freeze_modules and losses matches number of heads of the model
        if len(self._data["train"]["freeze_modules"]["heads"]) != n_heads:
            self._data["train"]["freeze_modules"]["heads"] = [False]*n_heads

        if len(self._data["train"]["losses"]["weights"]) != n_heads:
            self._data["train"]["losses"]["weights"] = [1]*n_heads

        # handle normalize augmentation - append to other augmentations
        if not("augmentations" in self._data["train"]["preprocessing"] and \
                self._data["train"]["preprocessing"]["augmentations"]):
            self._data["train"]["preprocessing"]["augmentations"] = []
        if self._data["train"]["preprocessing"]["normalize"]["use_normalize"]:
            self._data["train"]["preprocessing"]["augmentations"].append({
                "name":"Normalize",
                "params":self._data["train"]["preprocessing"]["normalize"]["params"] 
                    if "params" in self._data["train"]["preprocessing"]["normalize"] and \
                        self._data["train"]["preprocessing"]["normalize"]["params"] else {}
            })

        # handle accelerator
        if self._data["trainer"]["accelerator"] == "auto":
            accelerator = "gpu" if torch.cuda.is_available() else "cpu"
            self._data["trainer"]["accelerator"] = accelerator
            warnings.warn(f"Setting accelerator to '{accelerator}'.")