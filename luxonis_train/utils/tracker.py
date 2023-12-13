from luxonis_ml.tracker import LuxonisTracker
from pytorch_lightning.loggers.logger import Logger


class LuxonisTrackerPL(LuxonisTracker, Logger):
    """Implementation of LuxonisTracker that is compatible with PytorchLightning."""

    ...
