import argparse
from luxonis_train.core import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="Configuration file to use"
    )
    parser.add_argument(
        "--override", default=None, type=str, help="Manually override config parameter"
    )
    args = parser.parse_args()
    args_dict = vars(args)

    trainer = Trainer(args.config, args_dict)
    trainer.train()

    # Example: train in new thread
    # import time
    # trainer.train(new_thread=True)
    # while True:
    #     time.sleep(5)
    #     print(trainer.get_status())
    #     print(trainer.get_status_percentage(), trainer.get_save_dir())
