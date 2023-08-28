import torch
import os
import onnx
import onnxsim
import yaml
import pytorch_lightning as pl
from typing import Union, Optional
from pathlib import Path
from dotenv import load_dotenv

from luxonis_train.utils.config import Config
from luxonis_train.models import Model
from luxonis_train.models.heads import *
from luxonis_train.utils.filesystem import LuxonisFileSystem


class Exporter(pl.LightningModule):
    def __init__(self, cfg: Union[str, dict], args: Optional[dict] = None):
        """Main API which is used for exporting models trained with this library to .onnx, openVINO and .blob format.

        Args:
            cfg (Union[str, dict]): path to config file or config dict used to setup training
            args (Optional[dict]): argument dict provided through command line, used for config overriding
        """
        super().__init__()

        load_dotenv()

        self.cfg = Config(cfg)
        if args and args["override"]:
            self.cfg.override_config(args["override"])
        self.cfg.validate_config_exporter()

        # ensure save directory
        Path(self.cfg.get("exporter.export_save_directory")).mkdir(
            parents=True, exist_ok=True
        )

        self.model = Model()
        self.model.build_model()

        pretrained_path = self.cfg.get("exporter.export_weights")
        if pretrained_path:
            self.load_checkpoint(pretrained_path)

        self.model.eval()
        self.to_deploy()

    def load_checkpoint(self, path: str):
        """Loads checkpoint weights from provided path"""
        print(f"Loading weights from: {path}")
        fs = LuxonisFileSystem(path)
        checkpoint = torch.load(fs.read_to_byte_buffer())
        state_dict = checkpoint["state_dict"]
        # remove weights that are not part of the model
        removed = []
        for key in list(state_dict.keys()):
            if not key.startswith("model"):
                removed.append(key)
                state_dict.pop(key)
        if len(removed):
            print(f"Following weights weren't loaded: {removed}")

        self.load_state_dict(state_dict)

    def to_deploy(self):
        """Switch modules of the model to deploy"""
        for module in self.model.modules():
            if hasattr(module, "to_deploy"):
                module.to_deploy()

    def forward(self, inputs: torch.Tensor):
        """Forward function used in export"""
        outputs = self.model(inputs)
        return outputs

    def export(self):
        """Exports model to onnx and optionally to openVINO and .blob format"""
        dummy_input = torch.rand(1, 3, *self.cfg.get("exporter.export_image_size"))
        base_path = self.cfg.get("exporter.export_save_directory")
        output_names = self._get_output_names()

        print("Converting PyTorch model to ONNX")
        onnx_path = os.path.join(
            base_path, f"{self.cfg.get('exporter.export_model_name')}.onnx"
        )
        self.to_onnx(
            onnx_path,
            dummy_input,
            opset_version=self.cfg.get("exporter.onnx.opset_version"),
            input_names=["input"],
            output_names=output_names,
            dynamic_axes=self.cfg.get("exporter.onnx.dynamic_axes"),
        )
        model_onnx = onnx.load(onnx_path)
        onnx_model, check = onnxsim.simplify(model_onnx)
        if not check:
            raise RuntimeError("Onnx simplify failed.")
        onnx.save(onnx_model, onnx_path)

        if self.cfg.get("exporter.openvino.active"):
            import subprocess

            print("Converting ONNX to openVINO")
            output_list = ",".join(output_names)

            cmd = (
                f"mo --input_model {onnx_path} "
                f"--output_dir {base_path} "
                f"--model_name {self.cfg.get('exporter.export_model_name')} "
                f"--data_type {self.cfg.get('exporter.data_type')} "
                f"--scale_values '{self.cfg.get('exporter.scale_values')}' "
                f"--mean_values '{self.cfg.get('exporter.mean_values')}' "
                f"--output {output_list}"
            )

            if self.cfg.get("exporter.reverse_input_channels"):
                cmd += " --reverse_input_channels "

            subprocess.check_output(cmd, shell=True)

        if self.cfg.get("exporter.blobconverter.active"):
            import blobconverter

            print("Converting ONNX to .blob")

            optimizer_params = [
                f"--scale_values={self.cfg.get('exporter.scale_values')}",
                f"--mean_values={self.cfg.get('exporter.mean_values')}",
            ]
            if self.cfg.get("exporter.reverse_input_channels"):
                optimizer_params.append("--reverse_input_channels")

            blob_path = blobconverter.from_onnx(
                model=onnx_path,
                optimizer_params=optimizer_params,
                data_type=self.cfg.get("exporter.data_type"),
                shaves=self.cfg.get("exporter.blobconverter.shaves"),
                use_cache=False,
                output_dir=base_path,
            )

        print(f"Finished exporting. Files saved in: {base_path}")

        if self.cfg.get("exporter.upload.active"):
            self._upload_to_s3(onnx_path)

    def _get_output_names(self):
        """Gets output names for each head"""
        output_names = []
        for i, head in enumerate(self.model.heads):
            curr_output = head.get_output_names(i)
            if isinstance(curr_output, str):
                output_names.append(curr_output)
            else:
                output_names.extend(curr_output)
        return output_names

    def _upload_to_s3(self, onnx_path: str):
        """Uploads .pt, .onnx and current config.yaml to specified s3 bucket"""
        fs = LuxonisFileSystem(
            self.cfg.get("exporter.upload.upload_directory"), allow_local=False
        )
        print(f"Started upload to {fs.full_path()}...")

        # upload .ckpt file
        fs.put_file(
            local_path=self.cfg.get("exporter.export_weights"),
            remote_path=f"{self.cfg.get('exporter.export_model_name')}.ckpt",
        )
        # upload .onnx file
        fs.put_file(
            local_path=onnx_path,
            remote_path=f"{self.cfg.get('exporter.export_model_name')}.onnx",
        )
        # upload config.yaml
        self.cfg.save_data("config.yaml")  # create temporary file
        fs.put_file(local_path="config.yaml", remote_path="config.yaml")
        os.remove("config.yaml")  # delete temporary file

        # generate and upload export_config.yaml compatible with modelconverter
        full_remote_path = fs.full_path()
        onnx_path = os.path.join(
            full_remote_path, f"{self.cfg.get('exporter.export_model_name')}.onnx"
        )
        modelconverter_config = self._get_modelconverter_config(onnx_path)
        with open("config_export.yaml", "w+") as f:
            yaml.dump(modelconverter_config, f, default_flow_style=False)
        fs.put_file(local_path="config_export.yaml", remote_path="config_export.yaml")
        os.remove("config_export.yaml")  # delete temporary file

        print(f"Files upload finished")

    def _get_modelconverter_config(self, onnx_path: str):
        """Generates export config from input config that is
            compatible with Luxonis modelconverter tool

        Args:
            onnx_path (str): Path to .onnx model
        """
        out_config = {
            "input_model": onnx_path,
            "scale_values": self.cfg.get("exporter.scale_values"),
            "mean_values": self.cfg.get("exporter.mean_values"),
            "reverse_input_channels": self.cfg.get("exporter.reverse_input_channels"),
            "use_bgr": not self.cfg.get("train.preprocessing.train_rgb"),
            "input_shape": [1, 3] + self.cfg.get("exporter.export_image_size"),
            "data_type": self.cfg.get("exporter.data_type"),
            "output": [{"name": name} for name in self._get_output_names()],
            "meta": {"description": self.cfg.get("exporter.export_model_name")},
        }
        return out_config
