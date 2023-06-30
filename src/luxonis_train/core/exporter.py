import torch
import os
import onnx
import onnxsim
import pytorch_lightning as pl
from typing import Union, Optional
from pathlib import Path

from luxonis_train.utils.config import Config
from luxonis_train.models import Model
from luxonis_train.models.heads import *
from luxonis_train.utils.head_type import *


class Exporter(pl.LightningModule):
    def __init__(self, cfg: Union[str, dict], args: Optional[dict] = None):
        """ Main API which is used for exporting models trained with this library to .onnx, openVINO and .blob format.

        Args:
            cfg (Union[str, dict]): path to config file or config dict used to setup training
            args (Optional[dict]): argument dict provided through command line, used for config overriding
        """
        super().__init__()

        self.cfg = Config(cfg)
        if args and args["override"]:
            self.cfg.override_config(args["override"])
        self.cfg.validate_config_exporter()

        # ensure save directory
        Path(self.cfg.get("exporter.export_save_directory")).mkdir(parents=True, exist_ok=True)

        self.model = Model()
        self.model.build_model()

        self.load_checkpoint(self.cfg.get("exporter.export_weights"))
        self.model.eval()
        self.to_deploy()

    def load_checkpoint(self, path: str):
        """ Loads checkpoint weights from provided path """
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

    def forward(self, inputs: torch.Tensor):
        """ Forward function used in export """
        outputs = self.model(inputs)
        return outputs
    
    def export(self):
        """ Exports model to onnx and optionally to openVINO and .blob format """
        dummy_input = torch.rand(1,3,*self.cfg.get("exporter.export_image_size"))
        base_path = self.cfg.get("exporter.export_save_directory")
        output_names = self._get_output_names()

        print("Converting PyTorch model to ONNX")
        onnx_path = os.path.join(base_path, f"{self.cfg.get('exporter.export_model_name')}.onnx") 
        self.to_onnx(
            onnx_path,
            dummy_input,
            opset_version=self.cfg.get("exporter.onnx.opset_version"),
            input_names=["input"],
            output_names=output_names,
            dynamic_axes=self.cfg.get("exporter.onnx.dynamic_axes")
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

            cmd = f"mo --input_model {onnx_path} " \
            f"--output_dir {base_path} " \
            f"--model_name {self.cfg.get('exporter.export_model_name')} " \
            f"--data_type {self.cfg.get('exporter.data_type')} " \
            f"--scale_values '{self.cfg.get('exporter.scale_values')}' " \
            f"--mean_values '{self.cfg.get('exporter.mean_values')}' "  \
            f"--output {output_list}"

            if self.cfg.get("exporter.reverse_input_channels"):
                cmd += " --reverse_input_channels "

            subprocess.check_output(cmd, shell=True)
    
        if self.cfg.get("exporter.blobconverter.active"):
            import blobconverter
            print("Converting ONNX to .blob")

            optimizer_params=[
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
                output_dir=base_path
            )

        print(f"Finished exporting. Files saved in: {base_path}")

        if self.cfg.get("exporter.s3_upload.active"):
            self._upload_to_s3(onnx_path)

    def _get_output_names(self):
        """ Gets output names for each head """
        output_names = []
        for i, head in enumerate(self.model.heads):
            if isinstance(head, YoloV6Head):
                output_names.extend(["output1_yolov6r2", "output2_yolov6r2", "output3_yolov6r2"])
                if head.is_4head:
                    output_names.extend(["output4_yolov6r2"])
            elif isinstance(head.type, SemanticSegmentation):
                output_names.append("segmentation")
            else:
                output_names.append(f"output{i}")
        return output_names

    def _upload_to_s3(self, onnx_path):
        """ Uploads .pt, .onnx and current config.yaml to specified s3 bucket """
        import boto3
        from dotenv import load_dotenv

        load_dotenv()
        s3_client = boto3.client("s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            endpoint_url=os.getenv("AWS_S3_ENDPOINT_URL")
        )
        base_upload_key = f"{self.cfg.get('exporter.s3_upload.upload_directory')}/{self.cfg.get('exporter.export_model_name')}"
        
        # upload .ckpt file
        s3_client.upload_file(
            Filename=self.cfg.get("exporter.export_weights"),
            Bucket=self.cfg.get("exporter.s3_upload.bucket"),
            Key=f"{base_upload_key}/{self.cfg.get('exporter.export_model_name')}.ckpt")
        
        # upload .onnx file
        s3_client.upload_file(
            Filename=onnx_path,
            Bucket=self.cfg.get("exporter.s3_upload.bucket"),
            Key=f"{base_upload_key}/{self.cfg.get('exporter.export_model_name')}.onnx")
        
        # upload config.yaml
        self.cfg.save_data("config.yaml") # create temporary file
        s3_client.upload_file(
            Filename="config.yaml",
            Bucket=self.cfg.get("exporter.s3_upload.bucket"),
            Key=f"{base_upload_key}/config.yaml")
        os.remove("config.yaml") # delete temporary file

        print(f"Files uploaded to: {base_upload_key}")
        