from luxonis_ml.utils import Registry

BACKBONES = Registry(name="backbones")
NECKS = Registry(name="necks")
HEADS = Registry(name="heads")

LOSSES = Registry(name="losses")

CALLBACKS = Registry(name="callbacks")

OPTIMIZERS = Registry(name="optimizers")

SCHEDULERS = Registry(name="schedulers")

LOADERS = Registry(name="loaders")
