import pytest
import torch
from vdcnn import CHAR_IDS, Config, KMaxPool, VDCNN, encode, load, save, train

torch.set_num_threads(1)


def test_encoding_padding_unknown_and_case():
    values = encode(["A🙂", "abcdefghi"], 8)
    assert values.dtype == torch.long
    assert values[0].tolist() == [CHAR_IDS["a"], 1, 0, 0, 0, 0, 0, 0]
    assert values.shape == (2, 8)


def test_kmax_order_and_gradient():
    x = torch.tensor([[[3., 1., 5., 2.]]], requires_grad=True)
    selected = KMaxPool(2)(x)
    assert selected.tolist() == [[[3., 5.]]]
    selected.sum().backward()
    assert x.grad.tolist() == [[[1., 0., 1., 0.]]]


@pytest.mark.parametrize("depth", [9, 17, 29, 49])
def test_depth_logits_and_backward(depth):
    model = VDCNN(Config(classes=3, depth=depth, width=2, hidden=8, k=2))
    output = model(encode(["hello", "world"], 32))
    assert output.shape == (2, 3)
    assert sum(isinstance(layer, torch.nn.Conv1d) for layer in model.modules()) == depth
    torch.nn.CrossEntropyLoss()(output, torch.tensor([0, 2])).backward()
    assert torch.isfinite(model.embedding.weight.grad).all()


def test_training_and_checkpoint_roundtrip(tmp_path):
    texts = ["sunny day", "dark night", "bright sun", "moon stars"]
    config = Config(width=2, hidden=8, k=2)
    model, losses = train(texts, [0, 1, 0, 1], epochs=2, length=32, config=config)
    assert len(losses) == 2 and all(loss > 0 for loss in losses)
    path = tmp_path / "model.pt"
    save(model, path, 32)
    restored, length = load(path)
    torch.testing.assert_close(model(encode(texts, length)), restored(encode(texts, length)))
    with pytest.raises(FileExistsError):
        save(model, path, 32)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        encode([""])
    with pytest.raises(ValueError):
        VDCNN(Config(depth=10))
    with pytest.raises(ValueError):
        train(["a", "b"], [0, 2])
    with pytest.raises(ValueError):
        VDCNN()(encode(["short"], 8))
