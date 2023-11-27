import torchmetrics

from .luxonis_metric import LuxonisMetric


class TorchMetricWrapper(LuxonisMetric):
    Metric: type[torchmetrics.Metric]

    def __new__(cls, **_):
        cls.__doc__ = cls.Metric.__doc__
        return super().__new__(cls)

    def __init__(self, **kwargs):
        node_attributes = kwargs.pop("node_attributes", None)
        protocol = kwargs.pop("protocol", None)
        required_labels = kwargs.pop("required_labels", None)
        super().__init__(
            node_attributes=node_attributes,
            protocol=protocol,
            required_labels=required_labels,
        )
        self.metric = self.Metric(**kwargs)

    def update(self, *args, **kwargs):
        self.metric.update(*args, **kwargs)

    def compute(self):
        return self.metric.compute()


class Accuracy(TorchMetricWrapper):
    Metric = torchmetrics.Accuracy


class F1Score(TorchMetricWrapper):
    Metric = torchmetrics.F1Score


class JaccardIndex(TorchMetricWrapper):
    Metric = torchmetrics.JaccardIndex


class Precision(TorchMetricWrapper):
    Metric = torchmetrics.Precision


class Recall(TorchMetricWrapper):
    Metric = torchmetrics.Recall
