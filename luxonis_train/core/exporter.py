import torch
import os
import blobconverter
import torch.nn as nn
import pytorch_lightning as pl
import subprocess
import onnx
import onnxsim
from typing import Union
from pathlib import Path

from luxonis_train.utils.config import Config
from luxonis_train.models import Model
from luxonis_train.models.heads import *
from luxonis_train.utils.head_type import *


class Exporter(pl.LightningModule):
    def __init__(self, args: dict, cfg: Union[str, dict]):
        """ Main API which is used for exporting models trained with this library to .onnx, openVINO and .blob format.

        Args:
            args (dict): argument dict provided through command line, used for config overriding
            cfg (Union[str, dict]): path to config file or config dict used to setup training
        """
        super().__init__()

        self.cfg = Config(cfg)
        if args["override"]:
            self.cfg.override_config(args["override"])

        # ensure save directory
        Path(self.cfg.get("exporter.export_save_directory")).mkdir(parents=True, exist_ok=True)

        self.model = Model()
        self.model.build_model()

        self.load_checkpoint(self.cfg.get("exporter.export_weights"))
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
    
    def export(self):
        """ Export model to onnx, openVINO and .blob format """
        dummy_input = torch.rand(1,3,*self.cfg.get("exporter.export_image_size"))
        base_path = self.cfg.get("exporter.export_save_directory")
        output_names = self._get_output_names()

        print("Converting PyTorch model to ONNX")
        onnx_path = os.path.join(base_path, f"{self.cfg.get('model.name')}.onnx") 
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
        if not check:
            raise RuntimeError("Onnx simplify failed.")
        onnx.save(onnx_model, onnx_path)

        print("Converting ONNX to openVINO")
        output_list = ",".join(output_names)

        cmd = f"mo --input_model {onnx_path} " \
        f"--output_dir {base_path} " \
        f"--model_name {self.cfg.get('model.name')} " \
        "--reverse_input_channels " \
        "--data_type FP16 " \
        "--scale_values '[58.395, 57.120, 57.375]' " \
        "--mean_values '[123.675, 116.28, 103.53]' "  \
        f"--output {output_list}"

        subprocess.check_output(cmd, shell=True)
    
        print("Converting IR to blob")
        xmlfile = f"{os.path.join(base_path, self.cfg.get('model.name'))}.xml"
        binfile = f"{os.path.join(base_path, self.cfg.get('model.name'))}.bin"
        blob_path = blobconverter.from_openvino(
            xml=xmlfile,
            bin=binfile,
            data_type="FP16",
            shaves=6,
            version="2022.1",
            use_cache=False,
            output_dir=base_path
        )

        print(f"Finished exporting. Files saved in: {base_path}")

    def to_blob(self, remove_onnx = True):
        """ Export model from onnx to blob directly """
        dummy_input = torch.rand(1,3,*self.cfg.get("exporter.export_image_size"))
        base_path = self.cfg.get("exporter.export_save_directory")
        output_names = self._get_output_names()

        print("Converting PyTorch model to ONNX")
        onnx_path = os.path.join(base_path, f"{self.cfg.get('model.name')}.onnx") 
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
        if not check:
            raise RuntimeError("Onnx simplify failed.")
        onnx.save(onnx_model, onnx_path)
        
        print("Converting ONNX to blob")
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
