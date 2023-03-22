import os
import yaml
import warnings

DB_PATH = "./configs/db" # probably a nicer way to do this

def cfg_override(cfg, args):
    items = args.split(" ")
    if len(items) % 2 != 0:
        raise RuntimeError("Parameters passed by --override should be in 'key value' shape but one value is missing.")

    for i in range(0, len(items), 2):
        keys = items[i]
        value = items[i+1]
        subkeys = keys.split(".")
        sub_dict = cfg
        for key in subkeys[:-1]:
            if isinstance(sub_dict, list): # check if key should be list index 
                key = int(key)
            sub_dict = sub_dict[key]

        if subkeys[-1] not in sub_dict:
            raise RuntimeError(f"Key '{keys}' not present in config.")
        key = subkeys[-1] if isinstance(sub_dict, dict) else int(subkeys[-1])
        sub_dict[key] = int(value) if value.isdigit() else value
    return cfg

def check_cfg(cfg):
    # TODO: more checks, now only basic ones related to model creation

    model_cfg = cfg["model"]
    model_predefined = model_cfg["type"] != None
    backbone_specified = "backbone" in model_cfg and model_cfg["backbone"]
    neck_specified = "neck" in model_cfg and model_cfg["neck"]
    heads_specified = "heads" in model_cfg and isinstance(model_cfg["heads"], list)
    
    if model_predefined:
        if backbone_specified or neck_specified or heads_specified:
            warnings.warn("Settings of backbone/neck/heads will be overridden by predefined model.", SyntaxWarning)
        if not("n_classes" in model_cfg["params"] and model_cfg["params"]["n_classes"]):
            raise RuntimeError("Model 'n_classes' param must be defined if you use predefined model.")

    if not model_predefined:
        if not (backbone_specified and heads_specified):
            raise RuntimeError("Backbone and at least 1 head must be specified.")

        for head in model_cfg["heads"]:
            if not("n_classes" in head["params"] and head["params"]["n_classes"]):
                raise RuntimeError("Head 'n_classes' param must be defined for every head.")

        if "params" in model_cfg and model_cfg["params"]:
            warnings.warn("Model-wise parameters won't be taken into account if you don't specify model type.", SyntaxWarning)
        
        if "additional_heads" in model_cfg and model_cfg["additional_heads"]:
            warnings.warn("Additional heads won't be taken into account if you don't specify model type. \
                Move them to 'heads' if you want to use them.", SyntaxWarning)

        if model_cfg["pretrained"] and model_cfg["backbone"]["pretrained"]:
            warnings.warn("Weights of the backbone will be overridden by whole model weights.", SyntaxWarning)

    # check if image size specified
    if not("image_size" in cfg["train"] and cfg["train"]["image_size"]):
        warnings.warn("Image size not specified (under 'train'). Using default size [256, 256].")
        cfg["train"]["image_size"] = [256,256]
    
def check_cfg_export(cfg):
    if "export" not in cfg:
        raise RuntimeError("No 'export' section found in config file.")

    if not("weights" in cfg["export"] and cfg["export"]["weights"]):
        raise RuntimeError("No 'weights' speficied in config file.")
    
    if not("save_directory" in cfg["export"] and cfg["export"]["save_directory"]):
        warnings.warn("No save directory specified. Using default location 'output'")
        cfg["export"]["save_directory"] = cfg["export"]["source_directory"]

    if not("image_size" in cfg["export"] and cfg["export"]["image_size"]):
        warnings.warn("Image size not specified (under 'export'). Using default size [256, 256].")
        cfg["export"]["image_size"] = [256,256]

def load_predefined_cfg(cfg):
    model_type = cfg["model"]["type"]
    if model_type.startswith("YoloV6"):
        load_yolov6_cfg(cfg)
    else:
        raise RuntimeError(f"{model_type} not supported")

def load_yolov6_cfg(cfg):
    predefined_cfg_path = cfg["model"]["type"].lower() +".yaml"
    with open(os.path.join(DB_PATH, predefined_cfg_path), "r") as f:
        predefined_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    cfg["model"]["backbone"] = predefined_cfg["backbone"]
    cfg["model"]["neck"] = predefined_cfg["neck"]
    cfg["model"]["heads"] = predefined_cfg["heads"]
    
    model_params = cfg["model"]["params"] if "params" in cfg["model"]  and cfg["model"]["params"] else {}
    for key, value in model_params.items():
        if value is None:
            continue
        if key == "n_classes":
            cfg["model"]["heads"][0]["params"]["n_classes"] = value
            cfg["model"]["heads"][0]["loss"]["params"]["n_classes"] = value

    if "additional_heads" in cfg["model"] and isinstance(cfg["model"]["additional_heads"], list):
        cfg["model"]["heads"].extend(cfg["model"]["additional_heads"])