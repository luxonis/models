import os
import pytorch_lightning as pl
import threading
from dotenv import load_dotenv
from luxonis_ml import *
from pytorch_lightning.utilities import rank_zero_only

from luxonis_train.models import ModelLightningModule
from luxonis_train.utils.augmentations import TrainAugmentations, ValAugmentations

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
        self.rank = rank_zero_only.rank    

        train_cfg = cfg["train"]
        logger_cfg = cfg["logger"]
        epochs, batch_size, eval_interval = train_cfg["epochs"], train_cfg["batch_size"], train_cfg["eval_interval"]

        load_dotenv() # loads env variables for mlflow logging
        logger = LuxonisTrackerPL(rank=self.rank, **logger_cfg)
        logger.log_hyperparams({"epochs": epochs, "batch_size": batch_size, "accumulate_grad_batches": train_cfg["accumulate_grad_batches"]})
        run_save_dir = os.path.join(logger_cfg["save_directory"], logger.run_name)
        
        use_ddp = True if (args["devices"] == None or \
            isinstance(args["devices"], list) and len(args["devices"]) > 1 or \
            isinstance(args["devices"], int) and args["devices"]>1) \
            else False

        self.train_augmentations = None
        self.val_augmentations = None

        self.lightning_module = ModelLightningModule(self.cfg, run_save_dir)
        self.pl_trainer = pl.Trainer(
            accelerator=args["accelerator"],
            devices=args["devices"],
            strategy="ddp" if use_ddp else None,
            logger=logger,
            max_epochs=epochs,
            accumulate_grad_batches=train_cfg["accumulate_grad_batches"],
            check_val_every_n_epoch=eval_interval,
            num_sanity_val_steps=2,
        )

    @rank_zero_only
    def get_status(self):
        """Get current status of training

        Returns:
            Tuple(int, int): First element is current epoch, second element is total number of epochs
        """
        return self.lightning_module.get_status()
    
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
            else:
                self.thread = threading.Thread(
                    target=self.pl_trainer.fit,
                    args=(self.lightning_module, pytorch_loader_train, pytorch_loader_val),
                    daemon=True
                )
                self.thread.start()
