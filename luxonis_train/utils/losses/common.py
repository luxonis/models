import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(**kwargs)
    
    def forward(self, preds, labels, **kwargs):
        if labels.ndim == 4:
            # target should be of size (N,...)
            labels = labels.argmax(dim=1) 
        return self.criterion(preds, labels)

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, **kwargs):
        super(BCEWithLogitsLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, preds, labels, **kwargs):
        return self.criterion(preds, labels)