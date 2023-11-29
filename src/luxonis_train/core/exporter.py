import os
import os.path as osp
import tempfile
from pathlib import Path
from typing import Any

import onnx
import onnxsim
import yaml
from luxonis_ml.utils import LuxonisFileSystem
from torch import Size

from luxonis_train.models import LuxonisModel
from luxonis_train.utils.config import Config

from .core import Core


class Exporter(Core):
    """Main API which is used to create the model, setup pytorch lightning environment
    and perform training based on provided arguments and config."""

    def __init__(
        self,
        cfg: str | dict[str, Any] | Config,
        opts: list[str] | tuple[str, ...] | None = None,
    ):
        """Constructs a new Exporter instance.

        Args:
            cfg (str | dict): path to config file or config dict used to setup training
            args (dict | None): argument dict provided through command line, used for config overriding
        """

        super().__init__(cfg, opts)

        input_shape = self.cfg.exporter.input_shape
        if self.cfg.model.weights is None:
            raise ValueError(
                "Model weights must be specified in config file for export."
            )
        self.local_path = self.cfg.model.weights
        if input_shape is None:
            self.input_shape = self.loader_val.input_shape
        else:
            self.input_shape = Size(input_shape)

        self.export_path = osp.join(
            self.cfg.exporter.export_save_directory, self.cfg.exporter.export_model_name
        )

        normalize_params = self.cfg.train.preprocessing.normalize.params
        if self.cfg.exporter.scale_values is not None:
            self.scale_values = self.cfg.exporter.scale_values
        else:
            self.scale_values = normalize_params.get("std", None)

        if self.cfg.exporter.mean_values is not None:
            self.mean_values = self.cfg.exporter.mean_values
        else:
            self.mean_values = normalize_params.get("mean", None)

        self.lightning_module = LuxonisModel(
            cfg=self.cfg,
            save_dir=self.run_save_dir,
            input_shape=self.input_shape,
            dataset_metadata=self.dataset_metadata,
        )

    def _get_modelconverter_config(self, onnx_path: str) -> dict[str, Any]:
        """Generates export config from input config that is compatible with Luxonis
        modelconverter tool.

        Args:
            onnx_path (str): Path to .onnx model
        """
        return {
            "input_model": onnx_path,
            "scale_values": self.scale_values,
            "mean_values": self.mean_values,
            "reverse_input_channels": self.cfg.exporter.reverse_input_channels,
            "use_bgr": not self.cfg.train.preprocessing.train_rgb,
            "input_shape": list(self.input_shape),
            "data_type": self.cfg.exporter.data_type,
            "output": [{"name": name} for name in self.output_names],
            "meta": {"description": self.cfg.model.name},
        }

    def export(self, onnx_path: str | None = None):
        """Runs export."""
        onnx_path = onnx_path or self.export_path + ".onnx"
        self.output_names = self.lightning_module.export_onnx(
            onnx_path, **self.cfg.exporter.onnx.model_dump()
        )

        model_onnx = onnx.load(onnx_path)
        onnx_model, check = onnxsim.simplify(model_onnx)
        if not check:
            raise RuntimeError("Onnx simplify failed.")
        onnx.save(onnx_model, onnx_path)
        files_to_upload = [self.local_path, onnx_path]

        if self.cfg.exporter.blobconverter.active:
            import blobconverter

            print("Converting ONNX to .blob")

            optimizer_params = [
                f"--scale_values={self.scale_values}",
                f"--mean_values={self.mean_values}",
            ]
            if self.cfg.exporter.reverse_input_channels:
                optimizer_params.append("--reverse_input_channels")

            blob_path = blobconverter.from_onnx(
                model=onnx_path,
                optimizer_params=optimizer_params,
                data_type=self.cfg.exporter.data_type,
                shaves=self.cfg.exporter.blobconverter.shaves,
                use_cache=False,
                output_dir=self.export_path,
            )
            files_to_upload.append(blob_path)

        if self.cfg.exporter.upload.active:
            self._upload_to_s3(files_to_upload)

    def _upload_to_s3(self, files_to_upload: list[str]):
        """Uploads .pt, .onnx and current config.yaml to specified s3 bucket."""
        fs = LuxonisFileSystem(
            self.cfg.exporter.upload.upload_directory, allow_local=False
        )
        print(f"Started upload to {fs.full_path()}...")

        for file in files_to_upload:
            suffix = Path(file).suffix
            fs.put_file(
                local_path=file,
                remote_path=self.cfg.exporter.export_model_name + suffix,
            )

        with tempfile.TemporaryFile() as f:
            self.cfg.save_data(f.name)
            fs.put_file(local_path=f.name, remote_path="config.yaml")

        full_remote_path = fs.full_path()
        onnx_path = os.path.join(
            full_remote_path, f"{self.cfg.exporter.export_model_name}.onnx"
        )
        modelconverter_config = self._get_modelconverter_config(onnx_path)

        with tempfile.TemporaryFile() as f:
            yaml.dump(modelconverter_config, f, default_flow_style=False)
            fs.put_file(local_path=f.name, remote_path="config_export.yaml")

        print("Files upload finished")
