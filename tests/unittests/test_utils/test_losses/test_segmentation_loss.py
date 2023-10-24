import pytest
import torch
import torch.nn.functional as F
from luxonis_train.utils.losses.common import SegmentationLoss

# pytest.skip(allow_module_level=True)


# TODO: move to sep file as it may be useful
def make_one_hot(sparse_targets: torch.Tensor, n_classes: int) -> torch.Tensor:
    targets = (
        F.one_hot(sparse_targets, num_classes=n_classes)
        .to(torch.float32)
        .movedim(-1, 1)
        .contiguous()
    )
    return targets


def test_segmentation_loss_bce_loss():
    shape = (bs := 8, n_classes := 4, spatial := 16, spatial)

    loss_fn = SegmentationLoss(n_classes=n_classes)

    predictions = F.softmax(torch.rand(shape) * 10 - 5, dim=1).view(bs, -1)
    targets_sparse = torch.randint(0, n_classes, (bs, spatial, spatial))
    targets = make_one_hot(targets_sparse, n_classes).view(bs, -1)

    loss = loss_fn.bce(predictions, targets).mean(1)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([bs])


def test_segmentation_loss_focal_loss():
    shape = (bs := 8, n_classes := 4, spatial := 16, spatial)

    loss_fn = SegmentationLoss(n_classes=n_classes)
    predictions = F.softmax(torch.rand(shape) * 10 - 5, dim=1)
    targets_sparse = torch.randint(0, n_classes, (bs, spatial, spatial))
    targets = make_one_hot(targets_sparse, n_classes)

    # using clone as the function does as well
    loss = loss_fn.focal_loss(predictions.clone(), targets.clone())

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([bs])


def test_segmentation_loss_forward():
    shape = (bs := 8, n_classes := 4, spatial := 16, spatial)

    loss_fn = SegmentationLoss(n_classes=n_classes)

    predictions = torch.rand(shape) * 10 - 5
    targets_sparse = torch.randint(0, n_classes, (bs, spatial, spatial))
    targets = make_one_hot(targets_sparse, n_classes)
    loss = loss_fn(predictions, targets.contiguous())

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])


def test_segmentation_loss_all_correct():
    shape = (bs := 8, n_classes := 4, spatial := 16, spatial)

    loss_fn = SegmentationLoss(n_classes=n_classes)

    predictions = torch.zeros(shape, dtype=torch.float32)
    predictions[:, 1, :, :] = (
        torch.ones((bs, spatial, spatial), dtype=torch.float32) * 100.0
    )
    targets_sparse = torch.ones((bs, spatial, spatial), dtype=torch.int64)
    targets = make_one_hot(targets_sparse, n_classes)
    loss = loss_fn(predictions, targets.contiguous())

    assert torch.allclose(loss, torch.tensor(0.0))


def test_segmentation_loss_invalid_input_shape():
    shape = (bs := 8, n_classes := 4, spatial := 16)

    loss_fn = SegmentationLoss(n_classes=n_classes)

    predictions = torch.rand(shape) * 10 - 5
    targets_sparse = torch.randint(0, n_classes, (bs, spatial))
    targets = make_one_hot(targets_sparse, n_classes)

    with pytest.raises(IndexError):
        loss_fn(predictions, targets)


@pytest.mark.skip(
    reason="check the current state of the loss before allowing this test"
)
def test_one_class():
    shape = (bs := 8, n_classes := 1, spatial := 16, spatial)

    loss_fn = SegmentationLoss(n_classes=n_classes)

    predictions = torch.rand(shape) * 10 - 5
    targets_sparse = torch.randint(0, n_classes, (bs, spatial, spatial))
    targets = make_one_hot(targets_sparse, n_classes)

    with pytest.raises(ValueError):
        loss_fn(predictions, targets)


if __name__ == "__main__":
    pytest.main()
