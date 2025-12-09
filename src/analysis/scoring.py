import numpy as np
from typing import Tuple, List

def rank_heads(scores: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Given a 2D array of scores (layers x heads), return a list of
    (layer, head, score) tuples sorted by score in descending order.
    """
    ranked_heads = []
    n_layers, n_heads = scores.shape
    for layer in range(n_layers):
        for head in range(n_heads):
            ranked_heads.append((layer, head, scores[layer, head]))
    ranked_heads.sort(key=lambda x: x[2], reverse=True)
    return ranked_heads

def get_critical_heads(scores: np.ndarray, percentile: float = 90) -> List[Tuple[int, int]]:
    """
    Identify heads with scores above the given percentile.
    Returns a list of (layer, head) tuples.
    """
    threshold = np.percentile(scores, percentile)
    critical_heads = []
    n_layers, n_heads = scores.shape
    for layer in range(n_layers):
        for head in range(n_heads):
            if scores[layer, head] >= threshold:
                critical_heads.append((layer, head))
    return critical_heads

