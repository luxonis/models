import argparse
from pathlib import Path

from luxonis_train.core import Exporter, Inferer, Trainer, Tuner


def add_subparser(
    subparsers: argparse._SubParsersAction, name: str, help: str
) -> argparse.ArgumentParser:
    """Add a subparser to the parser."""
    parser = subparsers.add_parser(
        name, help=help, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, help="Path to the configuration file", required=True
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Additional options")

    return parser


# Create the top-level parser
parser = argparse.ArgumentParser(
    prog="luxonis_train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
subparsers = parser.add_subparsers(dest="command", help="commands")

add_subparser(subparsers, "train", "run the training")
eval_parser = add_subparser(subparsers, "eval", "run the evaluation")
eval_parser.add_argument(
    "--view",
    type=str,
    choices=["train", "val", "test"],
    default="val",
    help="View mode",
)
infer_parser = add_subparser(subparsers, "infer", "run the inference")
infer_parser.add_argument(
    "--view",
    type=str,
    choices=["train", "val", "test"],
    default="val",
    help="View mode",
)
add_subparser(subparsers, "tune", "run hyperparameter tuning")
add_subparser(subparsers, "export", "export the model")


def main():
    args = parser.parse_args()

    if args.command == "train":
        Trainer(str(args.config), args.opts).train()
    elif args.command == "eval":
        Trainer(str(args.config), args.opts).test(view=args.view)
    elif args.command == "tune":
        Tuner(str(args.config), args.opts).tune()
    elif args.command == "export":
        Exporter(str(args.config), args.opts).export()
    elif args.command == "infer":
        Inferer(str(args.config), args.opts, view=args.view).infer()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()