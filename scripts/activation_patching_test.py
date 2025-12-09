import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer, ActivationCache
from src.patching.activation_patching import run_activation_patching
from src.data.minimal_pair_gen import MinimalPair
from src.analysis.scoring import rank_heads, get_critical_heads

from src.analysis.visualization import plot_patching_heatmap

matplotlib.use("Agg")  # Use non-interactive backend for testing

model = HookedTransformer.from_pretrained("gpt2")

pair = MinimalPair(
    clean="The cats near the dog",
    corrupted="The cat near the dog"
)

patching_result = run_activation_patching(model, pair)

# Verify
assert patching_result.scores.shape == (12, 12)
assert patching_result.clean_logit_diff > 0 # plural should be favored
assert patching_result.corrupted_logit_diff < 0 # singular should be favored
print(f"Max score: {patching_result.scores.max().item():.4f} at {np.unravel_index(patching_result.scores.argmax(), patching_result.scores.shape)}")

pair = MinimalPair(
    clean="The cats near the dog",
    corrupted="The cat near the dog"
)

patching_result = run_activation_patching(model, pair)
percentile=95

# Verify
assert patching_result.scores.shape == (12, 12)
assert patching_result.clean_logit_diff > 0 # plural should be favored
assert patching_result.corrupted_logit_diff < 0 # singular should be favored
print(f"Max score: {patching_result.scores.max().item():.4f} at {np.unravel_index(patching_result.scores.argmax(), patching_result.scores.shape)}")

ranked = rank_heads(patching_result.scores)
print(f"Top 5 heads: {ranked[:5]}")

critical = get_critical_heads(patching_result.scores, percentile=percentile)
print(f"Critical heads (top {100-percentile}%): {critical}")

fig = plot_patching_heatmap(patching_result.scores, save_path="figures/test_activation_patching_heatmap.png")