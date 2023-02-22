import torch
import os
from pathlib import Path
import blobconverter

from luxonis_train.utils.config import *
from luxonis_train.models import Model


class Exporter:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        
        # check if model is predefined
        if self.cfg["model"]["type"]:
            self.load_predefined_cfg(self.cfg["model"]["type"])
        
        check_cfg_export(self.cfg)
        
        # ensure save directory
        Path(self.cfg["export"]["save_directory"]).mkdir(parents=True, exist_ok=True)

        self.model = Model()
        self.model.build_model(self.cfg["model"], self.cfg["export"]["image_size"])

        state_dict = torch.load(self.cfg["export"]["weights"], map_location="cpu")
        self.model.load_stete_dict(state_dict)

        # TODO: also need to switch to deploy parts of the model
        self.model.eval()

    
    def export(self):
        dummy_input = torch.rand(1,3,*self.cfg["export"]["image_size"])
        base_path = self.cfg["export"]["save_directory"]

        print("Converting PyTorch model to ONNX")
        torch.onnx.export(
            model=self.model,
            args=dummy_input,
            f=os.path.join(base_path, f"{self.cfg['train']['name']}.onnx"),
            opset_version=12,
            input_names=["input"],
            # output_names=[] #TODO
        )

        print("Converting ONNX to openVINO")
        cmd = f"mo --input_model {os.path.join(base_path, self.cfg['train']['name'])}.onnx " \
        f"--output_dir {base_path} " \
        f"--model_name {self.cfg['train']['name']} " \
        '--data_type FP16 ' \
        '--reverse_input_channel ' 
    
        print("Converting IR to blob")
        xmlfile = f"{os.path.join(base_path, self.cfg['train']['name'])}.xml"
        binfile = f"{os.path.join(base_path, self.cfg['train']['name'])}.bin"
        blob_path = blobconverter.from_openvino(
            xml=xmlfile,
            bin=binfile,
            data_type="FP16",
            shaves=6,
            output_dir=base_path
        )

