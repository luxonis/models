import argparse

from luxonis_train.core import Trainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Configuration file to use"
    )
    parser.add_argument(
        "--override", default=None, type=str, help="Manually override config parameter"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    trainer = Trainer(args.config, vars(args))
    trainer.train()
