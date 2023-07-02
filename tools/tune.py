import argparse
from luxonis_train.core import Tuner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, required=True, help="Configuration file to use")
    parser.add_argument("--override", default=None, type=str, help="Manually override config parameter")
    args = parser.parse_args()
    args_dict = vars(args)
    
    trainer = Tuner(args.config, args_dict)
    trainer.tune()