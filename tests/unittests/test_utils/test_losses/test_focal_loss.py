import pytest
import torch

from luxonis_train.utils.losses.common import FocalLoss

torch.manual_seed(42)


def test_focal_loss_min():
    loss_fn = FocalLoss(alpha=1.0, gamma=1, use_sigmoid=False)
    inputs = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0], requires_grad=True)
    targets = torch.tensor([0, 0, 1, 1, 0], dtype=torch.float32)

    loss = loss_fn(inputs, targets)

    assert torch.allclose(loss, torch.tensor(0.0))

def test_focal_loss_max():
    loss_fn = FocalLoss(alpha=1.0, gamma=1, use_sigmoid=False)
    inputs = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0], requires_grad=True)
    targets = torch.tensor([1, 1, 0, 0, 1], dtype=torch.float32)
    loss = loss_fn(inputs, targets)

    # torch clips negative infinity that log(0.0) would produce to -100; aslo exp{-100} is almost zero.
    assert torch.allclose(loss, torch.tensor(100.0))

def test_focal_loss_shapes():

    shapes = [(1, ), (1000, ), (1, 2, 2), (20, 256, 256), (64, 256, 256, 3)]
    
    loss_fn = FocalLoss(use_sigmoid=False)

    for shape in shapes:
        inputs = torch.rand(shape, requires_grad=True)
        targets = torch.randint(0, 2, shape)
        loss = loss_fn(inputs, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])  # Loss should be a scalar

def test_focal_loss_shapes_with_sigmoid():

    shapes = [(1, ), (1000, ), (1, 2, 2), (20, 256, 256), (64, 256, 256, 3)]
    
    loss_fn = FocalLoss(use_sigmoid=True)

    for shape in shapes:
        inputs = torch.rand(shape, requires_grad=True) * 100 - 50
        targets = torch.randint(0, 2, shape)
        loss = loss_fn(inputs, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])  # Loss should be a scalar

def test_focal_loss_sigmoid():
    shape = (8, 64, 64, 1)

    # values between -50 and 50
    inputs = torch.rand(shape, requires_grad=True) * 100 - 50
    targets = torch.randint(0, 2, shape)

    loss_fn_sig = FocalLoss(use_sigmoid=True)
    loss_fn_no_sig = FocalLoss(use_sigmoid=False)

    result_sig = loss_fn_sig(inputs, targets)
    result_no_sig = loss_fn_no_sig(torch.nn.functional.sigmoid(inputs), targets)
    assert torch.isclose(result_sig, result_no_sig)


def test_focal_loss_forgotten_sigmoid():

    loss_fn = FocalLoss(use_sigmoid=False)

    shape = (8, 64, 64, 1)

    # values between -50 and 50
    inputs = torch.rand(shape, requires_grad=True) * 100.0 - 50.0
    targets = torch.randint(0, 2, shape)

    with pytest.raises(RuntimeError):
        loss_fn(inputs, targets)


if __name__ == '__main__':
    pytest.main()
