from luxonis_train.utils.config import Config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--config', type=str, help="Path to training config", required=True)
parser.add_argument("--override", default="", type=str, help="Manually override config parameter")
args = parser.parse_args()

config = Config(args.config)
config.override_config(args.override)
# config.validate_config_exporter()
print(config)