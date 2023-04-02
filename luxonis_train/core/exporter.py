import torch
import os
from pathlib import Path
import blobconverter
import torch.nn as nn
import pytorch_lightning as pl
import subprocess
import onnx
import onnxsim

from luxonis_train.utils.config import *
from luxonis_train.models import Model
from luxonis_train.models.heads import *
from luxonis_train.utils.head_type import *


class Exporter(pl.LightningModule):
    def __init__(self, cfg: dict, args = {"override": None}):
        """ Main API which is used for exporting models trained with this library to .onnx, openVINO and .blob format.

        Args:
            cfg (dict): config dict used for exporting the model
        """
        super().__init__()
        self.cfg = cfg
        self.args = args
        
        if self.args["override"]:
            self.cfg = cfg_override(self.cfg, self.args["override"])

        # check if model is predefined
        if self.cfg["model"]["type"]:
            self.load_predefined_cfg(self.cfg["model"]["type"])
        
        check_cfg_export(self.cfg)
        
        # ensure save directory
        Path(self.cfg["export"]["save_directory"]).mkdir(parents=True, exist_ok=True)

        self.model = Model()
        self.model.build_model(self.cfg["model"], self.cfg["export"]["image_size"])

        self.load_checkpoint(self.cfg["export"]["weights"])

        self.model.eval()
        self.to_deploy()


    def load_checkpoint(self, path):
        """ Load checkpoint weights from provided path """
        print(f"Loading weights from: {path}")
        state_dict = torch.load(path)["state_dict"]
        # remove weights that are not part of the model
        removed = []
        for key in state_dict.keys():
            if not key.startswith("model"):
                removed.append(key)
                state_dict.pop(key)
        if len(removed):
            print(f"Following weights weren't loaded: {removed}")

        self.load_state_dict(state_dict)

    def to_deploy(self):
        """ Switch modules of the model to deploy"""
        for module in self.model.modules():
            if hasattr(module, "to_deploy"):
                module.to_deploy()

    def forward(self, inputs):
        """ Forward function used in export """
        outputs = self.model(inputs)
        return outputs
    
    def to_blob(self, remove_onnx = True):
        dummy_input = torch.rand(1,3,*self.cfg["export"]["image_size"])
        base_path = self.cfg["export"]["save_directory"]
        output_names = self._get_output_names()

        print("Converting PyTorch model to ONNX")
        onnx_path = os.path.join(base_path, f"{self.cfg['model']['name']}.onnx") 
        self.to_onnx(
            onnx_path,
            dummy_input,
            opset_version=12,
            input_names=["input"],
            output_names=output_names,
            dynamic_axes=None
        )
        model_onnx = onnx.load(onnx_path)
        onnx_model, check = onnxsim.simplify(model_onnx)
        assert check, "assert check failed"
        onnx.save(onnx_model, onnx_path)
        
        print("Converting ONNX to blob")
        xmlfile = f"{os.path.join(base_path, self.cfg['model']['name'])}.xml"
        binfile = f"{os.path.join(base_path, self.cfg['model']['name'])}.bin"
        blob_path = blobconverter.from_onnx(
            model=onnx_path,
            data_type="FP16",
            shaves=6,
            output_dir=base_path,
            use_cache=False
        )
        if remove_onnx:
            os.remove(onnx_path)
        print(f"Finished exporting. File saved in: {blob_path}")


    def export(self):
        """ Export model to onnx, openVINO and .blob format """
        dummy_input = torch.rand(1,3,*self.cfg["export"]["image_size"])
        base_path = self.cfg["export"]["save_directory"]
        output_names = self._get_output_names()

        print("Converting PyTorch model to ONNX")
        onnx_path = os.path.join(base_path, f"{self.cfg['model']['name']}.onnx") 
        self.to_onnx(
            onnx_path,
            dummy_input,
            opset_version=12,
            input_names=["input"],
            output_names=output_names,
            dynamic_axes=None
        )
        model_onnx = onnx.load(onnx_path)
        onnx_model, check = onnxsim.simplify(model_onnx)
        assert check, "assert check failed"
        onnx.save(onnx_model, onnx_path)

        print("Converting ONNX to openVINO")
        output_list = ",".join(output_names)

        cmd = f"mo --input_model {onnx_path} " \
        f"--output_dir {base_path} " \
        f"--model_name {self.cfg['model']['name']} " \
        "--data_type FP16 " \
        "--reverse_input_channel " 
        "--scale 255 " \
        f"--output {output_list} "

        subprocess.check_output(cmd, shell=True)
    
        print("Converting IR to blob")
        xmlfile = f"{os.path.join(base_path, self.cfg['model']['name'])}.xml"
        binfile = f"{os.path.join(base_path, self.cfg['model']['name'])}.bin"
        blob_path = blobconverter.from_openvino(
            xml=xmlfile,
            bin=binfile,
            data_type="FP16",
            shaves=6,
            version="2021.4",
            use_cache=False,
            output_dir=base_path
        )
        print(f"Finished exporting. Files saved in: {base_path}")

    def _get_output_names(self):
        """ Get output names for each head"""
        output_names = []
        for i, head in enumerate(self.model.heads):
            if isinstance(head, YoloV6Head):
                output_names.extend(["output1_yolov6r2", "output2_yolov6r2", "output3_yolov6r2"])
            elif isinstance(head.type, SemanticSegmentation):
                output_names.append("segmentation")
            else:
                output_names.append(f"output{i}")
        return output_names
