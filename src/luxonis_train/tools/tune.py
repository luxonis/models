from luxonis_train.core import Tuner

from .parser import make_parser


def get_args():
    parser = make_parser()
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    tuner = Tuner(args.config, args.opts)
    tuner.tune()
