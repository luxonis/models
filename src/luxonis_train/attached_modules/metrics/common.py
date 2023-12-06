import torchmetrics

from .base_metric import BaseMetric


class TorchMetricWrapper(BaseMetric):
    def __init__(self, **kwargs):
        node_attributes = kwargs.pop("node_attributes", None)
        protocol = kwargs.pop("protocol", None)
        required_labels = kwargs.pop("required_labels", None)
        super().__init__(
            node_attributes=node_attributes,
            protocol=protocol,
            required_labels=required_labels,
        )
        self.task = kwargs.get("task")
        if self.task == "multiclass":
            kwargs["num_classes"] = node_attributes["dataset_metadata"].n_classes
        elif self.task == "multilabel":
            kwargs["num_labels"] = node_attributes["dataset_metadata"].n_classes

        self.metric = self.Metric(**kwargs)

    def update(self, preds, target, *args, **kwargs):
        if self.task in ["multiclass"]:
            target = target.argmax(dim=1)
        self.metric.update(preds, target, *args, **kwargs)

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
