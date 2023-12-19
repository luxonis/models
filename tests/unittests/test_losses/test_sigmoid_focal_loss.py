import torch
from luxonis_train.attached_modules.losses import SigmoidFocalLoss

torch.manual_seed(42)


def test_forward_pass():
    batch_sizes = [1, 2, 10, 11, 15, 64, 128, 255]
    n_classes = [1, 2, 3, 4, 64]

    for bs in batch_sizes:
        for n_cl in n_classes:
            targets = torch.randint(0, 2, (bs, n_cl), dtype=torch.float32)
            predictions = torch.randn(bs, n_cl)
            loss_fn = SigmoidFocalLoss()

            loss = loss_fn.forward(predictions, targets)

            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0 or loss.shape == torch.Size([bs])


def test_different_alpha_gamma():
    bs, n_classes = 10, 4
    targets = torch.randint(0, 2, (bs, n_classes), dtype=torch.float32)
    predictions = torch.randn(bs, n_classes)

    for alpha in [0.25, 0.5, 0.75]:
        for gamma in [0.0, 1.0, 2.0]:
            loss_fn = SigmoidFocalLoss(alpha=alpha, gamma=gamma)
            loss = loss_fn.forward(predictions, targets)

            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0 or loss.shape == torch.Size([bs])


def test_reduction_methods():
    bs, n_classes = 10, 2
    targets = torch.randint(0, n_classes, (bs, 1), dtype=torch.float32)
    predictions = torch.rand(bs, 1)

    for reduction in ["none", "mean", "sum"]:
        loss_fn = SigmoidFocalLoss(reduction=reduction)  # type: ignore
        loss = loss_fn.forward(predictions, targets)

        if reduction == "none":
            assert loss.shape == torch.Size(
                [bs, 1]
            )  # NOTE: consistent with https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
        else:
            assert loss.dim() == 0
