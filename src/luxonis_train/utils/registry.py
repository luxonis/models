"""This module implements a metaclass for automatic registration of classes."""

from abc import ABCMeta

from luxonis_ml.utils.registry import Registry


class AutoRegisterMeta(ABCMeta):
    """Metaclass for automatically registering modules.

    Can be set as a metaclass for abstract base classes. Then, all subclasses will be
    automatically registered under the name of the subclass.

    Example:
        >>> REGISTRY = Registry(name="modules")
        >>> class BaseClass(metaclass=AutoRegisterMeta, registry=REGISTRY):
        ...     pass
        >>> class SubClass(BaseClass):
        ...     pass
        >>> REGISTRY["SubClass"]
        <class '__main__.SubClass'>
    """

    REGISTRY: Registry

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        attrs: dict[str, type],
        register: bool = True,
        register_name: str | None = None,
        registry: Registry | None = None,
    ):
        """Automatically register the class.

        Args:
            name (str): Class name
            bases (tuple): Base classes
            attrs (dict): Class attributes
            register (bool, optional): Weather to register the class. Defaults to True.
              Should be set to False for abstract base classes.
            register_name (str | None, optional): Name used for registration.
              If unset, the class name is used. Defaults to None.
            registry (Registry | None, optional): Registry to use for registration.
              Defaults to None. Has to be set in the base class.
        """
        new_class = super().__new__(cls, name, bases, attrs)
        if not hasattr(new_class, "REGISTRY"):
            if registry is not None:
                new_class.REGISTRY = registry
            elif register:
                raise ValueError(
                    "Registry has to be set in the base class or passed as an argument."
                )
        if register:
            (registry or new_class.REGISTRY).register_module(
                name=register_name or name, module=new_class
            )
        return new_class


CALLBACKS = Registry(name="callbacks")
LOADERS = Registry(name="loaders")
LOSSES = Registry(name="losses")
METRICS = Registry(name="metrics")
MODELS = Registry(name="models")
NODES = Registry(name="nodes")
OPTIMIZERS = Registry(name="optimizers")
SCHEDULERS = Registry(name="schedulers")
VISUALIZERS = Registry(name="visualizers")
