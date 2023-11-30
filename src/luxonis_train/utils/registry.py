"""This module implements a metaclass for automatic registration of classes."""


from luxonis_ml.utils.registry import Registry

CALLBACKS = Registry(name="callbacks")
LOADERS = Registry(name="loaders")
LOSSES = Registry(name="losses")
METRICS = Registry(name="metrics")
MODELS = Registry(name="models")
NODES = Registry(name="nodes")
OPTIMIZERS = Registry(name="optimizers")
SCHEDULERS = Registry(name="schedulers")
VISUALIZERS = Registry(name="visualizers")
