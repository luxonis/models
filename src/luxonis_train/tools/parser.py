import argparse


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Configuration file to use"
    )
    parser.add_argument(
        "opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Manually override config parameter",
    )
    return parser
