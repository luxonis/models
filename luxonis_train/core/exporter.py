import torch
import os
from pathlib import Path
import blobconverter
import torch.nn as nn
import pytorch_lightning as pl
import subprocess

from luxonis_train.utils.config import *
from luxonis_train.models import Model


class Exporter(pl.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()

        self.cfg = cfg
        
        # check if model is predefined
        if self.cfg["model"]["type"]:
            self.load_predefined_cfg(self.cfg["model"]["type"])
        
        check_cfg_export(self.cfg)
        
        # ensure save directory
        Path(self.cfg["export"]["save_directory"]).mkdir(parents=True, exist_ok=True)

        self.model = Model()
        self.model.build_model(self.cfg["model"], self.cfg["export"]["image_size"])

        self.load_checkpoint(self.cfg["export"]["weights"])

        # TODO: also need to switch to deploy parts of the model
        self.model.eval()


    def load_checkpoint(self, path):
        print(f"Loading weights from: {path}")
        state_dict = torch.load(path)["state_dict"]
        # remove weights from loss or other modules
        removed = []
        for key in state_dict.keys():
            if not key.startswith("model"):
                removed.append(key)
                state_dict.pop(key)
        if len(removed):
            print(f"Following weights weren't loaded: {removed}")

        self.load_state_dict(state_dict)

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs

    def export(self):
        dummy_input = torch.rand(1,3,*self.cfg["export"]["image_size"])
        base_path = self.cfg["export"]["save_directory"]

        print("Converting PyTorch model to ONNX")
        self.to_onnx(
            os.path.join(base_path, f"{self.cfg['model']['name']}.onnx"),
            dummy_input,
            opset_version=12,
            input_names=["input"],
            # output_names=[] TODO:
        )

        print("Converting ONNX to openVINO")
        cmd = f"mo --input_model {os.path.join(base_path, self.cfg['model']['name'])}.onnx " \
        f"--output_dir {base_path} " \
        f"--model_name {self.cfg['model']['name']} " \
        '--data_type FP16 ' \
        '--reverse_input_channel ' 
        subprocess.check_output(cmd, shell=True)
    
        print("Converting IR to blob")
        xmlfile = f"{os.path.join(base_path, self.cfg['model']['name'])}.xml"
        binfile = f"{os.path.join(base_path, self.cfg['model']['name'])}.bin"
        blob_path = blobconverter.from_openvino(
            xml=xmlfile,
            bin=binfile,
            data_type="FP16",
            shaves=6,
            output_dir=base_path
        )
        print(f"Finished exporting. Files saved in: {base_path}")

