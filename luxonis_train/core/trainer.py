import os
import pytorch_lightning as pl
import threading
from dotenv import load_dotenv
from luxonis_ml import *
from pytorch_lightning.utilities import rank_zero_only

from luxonis_train.models import ModelLightningModule
from luxonis_train.utils.config import *
from luxonis_train.utils.augmentations import TrainAugmentations, ValAugmentations
from luxonis_train.utils.general import get_current_label
from luxonis_train.utils.head_type import *

class Trainer:
    def __init__(self, args: dict, cfg: dict):
        """Main API which is used to create the model, setup pytorch lightning environment 
        and perform training based on provided arguments and config.

        Args:
            args (dict): argument dict provided through command line, control devices used to train and type of accelerator
            cfg (dict): configs dict used to setup training
        """
        
        self.args = args
        self.cfg = cfg
        if self.args["override"]:
            self.cfg = cfg_override(self.cfg, self.args["override"])

        # check if model is predefined
        if self.cfg["model"]["type"]:
            load_predefined_cfg(self.cfg)
        
        self._validate_dataset()
        check_cfg(self.cfg)
        
        self.rank = rank_zero_only.rank    

        train_cfg = cfg["train"]
        logger_cfg = cfg["logger"]
        epochs, batch_size, eval_interval = train_cfg["epochs"], train_cfg["batch_size"], train_cfg["eval_interval"]

        load_dotenv() # loads env variables for mlflow logging
        logger = LuxonisTrackerPL(rank=self.rank, **logger_cfg)
        logger.log_hyperparams({"epochs": epochs, "batch_size": batch_size, "accumulate_grad_batches": train_cfg["accumulate_grad_batches"]})
        self.run_save_dir = os.path.join(logger_cfg["save_directory"], logger.run_name)
        
        use_ddp = True if (args["devices"] == None or \
            isinstance(args["devices"], list) and len(args["devices"]) > 1 or \
            isinstance(args["devices"], int) and args["devices"]>1) \
            else False

        self.train_augmentations = None
        self.val_augmentations = None
        
        self.lightning_module = ModelLightningModule(self.cfg, self.run_save_dir)
        self.pl_trainer = pl.Trainer(
            accelerator=args["accelerator"],
            devices=args["devices"],
            strategy="ddp_find_unused_parameters_false" if use_ddp else None,
            logger=logger,
            max_epochs=epochs,
            accumulate_grad_batches=train_cfg["accumulate_grad_batches"],
            check_val_every_n_epoch=eval_interval,
            num_sanity_val_steps=2,
            #profiler="pytorch" # for debugging purposes
        )

        self.error_message = None

    def override_loss(self, custom_loss: object, head_id: int):
        """ Overrides loss function for specific head_id with custom loss """
        assert head_id in list(range(len(self.lightning_module.model.heads))), "head_id out of range"
        self.lightning_module.losses[head_id] = custom_loss

    def override_train_augmentations(self, aug: object):
        """ Overrides augmentations used for trainig dataset """
        self.train_augmentations = aug

    def override_val_augmentations(self, aug):
        """ Overrides augmentations used for validation dataset """
        self.val_augmentations = aug

    def run(self, new_thread: bool = False):
        """ Runs training

        Args:
            new_thread (bool, optional): Runs training in new thread if set to True. Defaults to False.
        """

        with LuxonisDataset(
            local_path=self.cfg["dataset"]["local_path"] if "local_path" in self.cfg["dataset"] else None,
            s3_path=self.cfg["dataset"]["s3_path"] if "s3_path" in self.cfg["dataset"] else None
        ) as dataset:

            if self.train_augmentations == None:
                self.train_augmentations = TrainAugmentations(
                    cfg=self.cfg["augmentations"] if self.cfg["augmentations"] else None,
                    image_size=self.cfg["train"]["image_size"]
                )
            
            loader_train = LuxonisLoader(dataset, view='train')
            loader_train.map(loader_train.auto_preprocess)
            loader_train.map(self.train_augmentations)
            pytorch_loader_train = loader_train.to_pytorch(
                batch_size=self.cfg["train"]["batch_size"],
                num_workers=self.cfg["train"]["n_workers"]
            )

            if self.val_augmentations == None:
                self.val_augmentations = ValAugmentations(image_size=self.cfg["train"]["image_size"])

            loader_val = LuxonisLoader(dataset, view="val")
            loader_val.map(loader_val.auto_preprocess)
            loader_val.map(self.val_augmentations)
            pytorch_loader_val = loader_val.to_pytorch(
                batch_size=self.cfg["train"]["batch_size"],
                num_workers=self.cfg["train"]["n_workers"]
            )

            if not new_thread:
                self.pl_trainer.fit(self.lightning_module, pytorch_loader_train, pytorch_loader_val)
                print(f"Checkpoints saved in: {self.get_save_dir()}")
            else:
                # Every time expection happens in the Thread, this hook will activate
                def thread_exception_hook(args):
                    self.error_message = str(args.exc_value)
                threading.excepthook = thread_exception_hook

                self.thread = threading.Thread(
                    target=self.pl_trainer.fit,
                    args=(self.lightning_module, pytorch_loader_train, pytorch_loader_val),
                    daemon=True
                )
                self.thread.start()
            
    @rank_zero_only
    def get_status(self):
        """Get current status of training

        Returns:
            Tuple(int, int): First element is current epoch, second element is total number of epochs
        """
        return self.lightning_module.get_status()
    
    @rank_zero_only
    def get_status_percentage(self):
        """ Return percentage of current training, takes into account early stopping

        Returns:
            float: Percentage of current training in range 0-100
        """
        return self.lightning_module.get_status_percentage()       

    @rank_zero_only
    def get_save_dir(self):
        """ Return path to directory where checkpoints are saved

        Returns:
            str: Save directory path
        """
        return self.run_save_dir
    
    @rank_zero_only
    def get_error_message(self):
        """ Return error message if one occures while running in thread, otherwise None

        Returns:
            str or None: Error message
        """
        return self.error_message
    
    @rank_zero_only
    def get_min_val_checkpoint_path(self):
        """ Return best min_val checkpoint path
        """
        return self.pl_trainer.checkpoint_callbacks[0].best_model_path
    
    @rank_zero_only
    def get_best_metric_checkpoint_path(self):
        """ Return best best_metric checkpoint path
        """
        return self.pl_trainer.checkpoint_callbacks[1].best_model_path
    
    def test(self, new_thread: bool = False):
        """ Runs testing
        Args:
            new_thread (bool, optional): Runs training in new thread if set to True. Defaults to False.
        """

        with LuxonisDataset(
            local_path=self.cfg["dataset"]["local_path"] if "local_path" in self.cfg["dataset"] else None,
            s3_path=self.cfg["dataset"]["s3_path"] if "s3_path" in self.cfg["dataset"] else None
        ) as dataset:

            self.test_augmentations = ValAugmentations(image_size=self.cfg["train"]["image_size"])

            loader_test = LuxonisLoader(dataset, view="test")
            loader_test.map(loader_test.auto_preprocess)
            loader_test.map(self.test_augmentations)
            pytorch_loader_test = loader_test.to_pytorch(
                batch_size=self.cfg["train"]["batch_size"],
                num_workers=self.cfg["train"]["n_workers"]
            )

            if not new_thread:
                self.pl_trainer.test(self.lightning_module, pytorch_loader_test)
            else:
                self.thread = threading.Thread(
                    target=self.pl_trainer.test,
                    args=(self.lightning_module, pytorch_loader_test),
                    daemon=True
                )
                self.thread.start()

    def _validate_dataset(self):
        """ Checks if number of classes specified in config file matches number of classes present in dataset.
        If it doesn't match we adopt value from dataset and override the config.
        """
        with LuxonisDataset(
            local_path=self.cfg["dataset"]["local_path"] if "local_path" in self.cfg["dataset"] else None,
            s3_path=self.cfg["dataset"]["s3_path"] if "s3_path" in self.cfg["dataset"] else None
        ) as dataset:
            dataset_n_classes = len(dataset.classes)
            
            # TODO: implement per task number of classes
            # for key in dataset.classes_by_task:
            #     print(key, len(dataset.classes_by_task[key]))

            model_cfg = self.cfg["model"]
            for head in model_cfg["heads"]:
                if head["params"] is None:
                    head["params"] = {}

                curr_n_classes = head["params"].get("n_classes", None)
                if curr_n_classes is None:
                    warnings.warn(f"Inheriting 'n_classes' parameter from dataset. Setting it to {dataset_n_classes}")
                elif curr_n_classes != dataset_n_classes:
                    warnings.warn(f"Number of classes in config ({curr_n_classes}) doesn't match number of \
                        classes in dataset ({dataset_n_classes}). Setting it to {dataset_n_classes}")
                head["params"]["n_classes"] = dataset_n_classes

                # also set n_classes to loss params (for now only if it's YoloV6 loss) 
                # TODO: avoid hardcoding, make the loss classes general so they can accept n_classes even if not used
                if head["loss"]["name"] == "YoloV6Loss":
                    if head["loss"]["params"] is None:
                        head["loss"]["params"] = {}
                    head["loss"]["params"]["n_classes"] = dataset_n_classes

    def _validate_dataset_old(self, loader):
        """ Checks if number of classes specified in config file matches number of classes present in annotations.

        Args:
            loader : Loader used for comparison

        Raises:
            RuntimeError
        """
        n_classes_dict = self.lightning_module.get_n_classes()
        _, labels = next(iter(loader))
        if "class" in n_classes_dict:
            curr_labels = get_current_label(Classification(), labels)
            n_classes_anno = curr_labels.shape[1]
            if n_classes_anno != n_classes_dict["class"]:
                raise RuntimeError(f"Number of classes in 'class' anotations ({n_classes_anno}) doesn't match" +
                    f"'n_classes' specified in config file ({n_classes_dict['class']})")
        if "segmentation" in n_classes_dict:
            curr_labels = get_current_label(SemanticSegmentation(), labels)
            n_classes_anno = curr_labels.shape[1]
            if n_classes_anno != n_classes_dict["segmentation"]:
                raise RuntimeError(f"Number of classes in 'segmentation' anotations ({n_classes_anno}) doesn't match" +
                    f"'n_classes' specified in config file ({n_classes_dict['segmentation']})")

