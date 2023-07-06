# TODO: every head has specific type, need information for correct metric calculation and callback

class Classification():
    def __init__(self):
        self._main_metric = "f1"
    
    @property
    def main_metric(self):
        return self._main_metric 
        
class MultiLabelClassification():
    def __init__(self):
        self._main_metric = "f1"
    
    @property
    def main_metric(self):
        return self._main_metric 

class ObjectDetection():
    def __init__(self):
        self._main_metric = "map"
    
    @property
    def main_metric(self):
        return self._main_metric

class SemanticSegmentation():
    def __init__(self):
        self._main_metric = "mIoU"
    
    @property
    def main_metric(self):
        return self._main_metric

class InstanceSegmentation():
    def __init__(self):
        self._main_metric = "mIoU"
    
    @property
    def main_metric(self):
        return self._main_metric    
        
class KeyPointDetection():
    def __init__(self):
        self._main_metric = "oks"
    
    @property
    def main_metric(self):
        return self._main_metric
