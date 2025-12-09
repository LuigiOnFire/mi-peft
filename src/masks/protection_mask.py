from typing import List, Tuple
import numpy as np

def mask_to_dict(critical_heads: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    """
    Convert list of (layer, head) tuples to a dictionary mapping layer -> list of heads
    """
    mask_dict = {}
    for layer, head in critical_heads:
        if layer not in mask_dict:
            mask_dict[layer] = []
        mask_dict[layer].append(head)
    return mask_dict

def compute_mask_coverage(critical_heads, n_layers, n_heads) -> float:
    """
    Compute the coverage of the critical heads mask as a fraction of total heads
    """
    total_heads = n_layers * n_heads
    covered_heads = len(critical_heads)
    return covered_heads / total_heads