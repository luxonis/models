import math

import pytest
import torch
from luxonis_train.attached_modules.losses import CrossEntropyLoss

torch.manual_seed(42)


def test_forward_pass():
    batch_sizes = [1, 2, 10, 11, 15, 64, 128, 255]
    n_classes = [1, 2, 3, 4, 64]

    for bs in batch_sizes:
        for n_cl in n_classes:
            targets = torch.randint(0, n_cl, (bs,), dtype=torch.int64)
            predictions = torch.randn(bs, n_cl)  # Logits for each class
            loss_fn = CrossEntropyLoss()

            loss = loss_fn.forward(predictions, targets)

            assert isinstance(loss, torch.Tensor)
            assert loss.shape == torch.Size([])


def test_max_loss():
    for n_classes in [2, 4, 10, 64]:
        bs, n_classes = 10, 4

        targets = torch.randint(0, n_classes, (bs,), dtype=torch.int64)
        predictions = torch.full((bs, n_classes), 1.0 / n_classes)

        loss_fn = CrossEntropyLoss()
        loss = loss_fn.forward(predictions, targets)
        max_loss = -math.log(1.0 / n_classes)

        assert torch.isclose(loss, torch.tensor(max_loss), atol=1e-6)


def test_min_loss():
    for n_classes in [2, 4, 10, 64]:
        bs, n_classes = 10, 4

        targets = torch.randint(0, n_classes, (bs,), dtype=torch.int64)
        predictions = torch.zeros((bs, n_classes))
        for i, target in enumerate(targets):
            predictions[i, target] = 10e3

        loss_fn = CrossEntropyLoss()
        loss = loss_fn.forward(predictions, targets)
        min_loss = 0.0

        assert torch.isclose(loss, torch.tensor(min_loss), atol=1e-6)


def test_weights():
    bs, n_classes = 10, 4
    targets = torch.randint(0, n_classes, (bs,), dtype=torch.int64)
    predictions = torch.randn(bs, n_classes)

    weights = torch.rand(n_classes)
    loss_fn_weight = CrossEntropyLoss(weight=weights.tolist())
    loss_fn_no_weight = CrossEntropyLoss()

    loss_weight = loss_fn_weight.forward(predictions, targets)
    loss_no_weight = loss_fn_no_weight.forward(predictions, targets)
    assert loss_weight != loss_no_weight


if __name__ == "__main__":
    pytest.main()
