import torch
from luxonis_train.attached_modules.losses import SoftmaxFocalLoss

torch.manual_seed(42)


def test_forward_pass():
    batch_sizes = [1, 2, 10, 11, 15, 64, 128, 255]
    n_classes = [1, 2, 3, 4, 64]

    for bs in batch_sizes:
        for n_cl in n_classes:
            targets = torch.randint(0, n_cl, (bs,), dtype=torch.int64)
            predictions = torch.randn(bs, n_cl)
            loss_fn = SoftmaxFocalLoss()

            loss = loss_fn.forward(predictions, targets)

            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0 or loss.shape == torch.Size([])


def test_different_alpha_gamma():
    bs, n_classes = 10, 4
    targets = torch.randint(0, n_classes, (bs,), dtype=torch.int64)
    predictions = torch.randn(bs, n_classes)

    alphas = [0.25, 0.5, [0.1, 0.2, 0.3, 0.4]]  # Including list of alphas
    gammas = [0.0, 1.0, 2.0]

    for alpha in alphas:
        for gamma in gammas:
            loss_fn = SoftmaxFocalLoss(alpha=alpha, gamma=gamma)
            loss = loss_fn.forward(predictions, targets)

            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0 or loss.shape == torch.Size([bs])


def test_reduction_methods():
    bs, n_classes = 10, 4
    targets = torch.randint(0, n_classes, (bs,), dtype=torch.int64)
    predictions = torch.randn((bs, n_classes))

    for reduction in ["none", "mean", "sum"]:
        loss_fn = SoftmaxFocalLoss(reduction=reduction)  # type: ignore
        loss = loss_fn.forward(predictions, targets)

        if reduction == "none":
            assert (
                loss.dim() == predictions.dim() - 1
            )  # NOTE: consistent with https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        else:
            assert loss.dim() == 0
