from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

from luxonis_train.core import Exporter, Inferer, Trainer, Tuner

app = typer.Typer(help="Luxonis Train CLI", add_completion=False)


class View(str, Enum):
    train = "train"
    val = "val"
    test = "test"

    def __str__(self):
        return self.value


ConfigType = Annotated[
    Optional[Path],
    typer.Option(
        help="Path to the configuration file.",
        show_default=False,
    ),
]

OptsType = Annotated[
    Optional[list[str]],
    typer.Argument(
        help="A list of optional CLI overrides of the config file.",
        show_default=False,
    ),
]

ViewType = Annotated[
    View,
    typer.Option(
        help="Which dataset view to use.",
    ),
]


@app.command()
def train(config: ConfigType = None, opts: OptsType = None):
    """Start training."""
    Trainer(str(config), opts).train()


@app.command()
def eval(config: ConfigType = None, view: ViewType = View.val, opts: OptsType = None):
    """Evaluate model."""
    Trainer(str(config), opts).test(view=view.name)


@app.command()
def tune(config: ConfigType = None, opts: OptsType = None):
    """Start hyperparameter tuning."""
    Tuner(str(config), opts).tune()


@app.command()
def export(config: ConfigType = None, opts: OptsType = None):
    """Export model."""
    Exporter(str(config), opts).export()


@app.command()
def infer(config: ConfigType = None, view: ViewType = View.val, opts: OptsType = None):
    """Run inference."""
    Inferer(str(config), opts, view=view.name).infer()


def main():
    app()


if __name__ == "__main__":
    main()
