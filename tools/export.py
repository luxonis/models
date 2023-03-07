import argparse
import yaml
from luxonis_train.core import Exporter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config', type=str, help="Path to training config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    exporter = Exporter(cfg)
    exporter.export()