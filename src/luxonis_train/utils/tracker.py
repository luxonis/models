from pytorch_lightning.loggers.logger import Logger
from luxonis_ml.utils import LuxonisTracker


class LuxonisTrackerPL(LuxonisTracker, Logger):
    """Implementation of LuxonisTracker that is compatible with PytorchLightning"""

    pass
