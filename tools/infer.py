import argparse
import json
from luxonis_train.core import Inferer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="Configuration file to use"
    )
    parser.add_argument(
        "--override",
        type=json.loads,
        help="Manually override config parameter, input in json format",
    )
    args = parser.parse_args()
    args_dict = vars(args)

    inferer = Inferer(args.config, args_dict)
    inferer.infer()
