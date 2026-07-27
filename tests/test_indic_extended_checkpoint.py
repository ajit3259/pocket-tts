import torch

from experiments.indic.prepare_extended_checkpoint import (
    expand_embedding,
    initialize_matched_random,
)


def test_matched_random_initialization_is_deterministic_and_nonzero() -> None:
    source = torch.arange(42, dtype=torch.float32).reshape(7, 6) / 10
    trained_ids = [1, 2, 3, 4]

    first = initialize_matched_random(source, trained_ids, 5, seed=123)
    second = initialize_matched_random(source, trained_ids, 5, seed=123)

    assert torch.equal(first, second)
    assert first.shape == (5, 6)
    assert torch.all(first.norm(dim=1) > 0)


def test_expand_embedding_preserves_tokens_and_moves_padding() -> None:
    source = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    initialized = torch.tensor([[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]])

    expanded = expand_embedding(source, initialized, preserved_vocab_size=4)

    assert expanded.shape == (7, 3)
    assert torch.equal(expanded[:4], source[:4])
    assert torch.equal(expanded[4:6], initialized)
    assert torch.equal(expanded[6], source[4])
