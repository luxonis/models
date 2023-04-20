from luxonis_train.utils.config import Config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--config', type=str, help="Path to training config", required=True)
args = parser.parse_args()

test = Config(args.config)
print(test)