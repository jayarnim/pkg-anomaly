import torch.nn as nn
from collections.abc import Iterator


def fc_block(
    input_dim: int, 
    hidden_dim: list[int], 
    dropout: float,
) -> Iterator[nn.Sequential]:
    IN_FEATRUES = input_dim
    
    for OUT_FEATURES in hidden_dim:
        yield nn.Sequential(
            nn.Linear(IN_FEATRUES, OUT_FEATURES),
            nn.LayerNorm(OUT_FEATURES),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        IN_FEATRUES = OUT_FEATURES