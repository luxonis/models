import torchmetrics

from luxonis_train.metrics.luxonis_metric import LuxonisMetric

# TODO: this is hacky and not optimal at all, could be automated for all torchmetrics


class Accuracy(LuxonisMetric, torchmetrics.Accuracy):
    def __new__(cls, **kwargs):
        instance = super().__new__(cls, **kwargs)
        instance.preprocess = LuxonisMetric._default_preprocess
        instance.name = "Accuracy"
        return instance


class F1Score(LuxonisMetric, torchmetrics.F1Score):
    def __new__(cls, **kwargs):
        instance = super().__new__(cls, **kwargs)
        instance.preprocess = LuxonisMetric._default_preprocess
        instance.name = "F1Score"
        return instance


class JaccardIndex(LuxonisMetric, torchmetrics.JaccardIndex):
    def __new__(cls, **kwargs):
        instance = super().__new__(cls, **kwargs)
        instance.preprocess = LuxonisMetric._default_preprocess
        instance.name = "JaccardIndex"
        return instance
