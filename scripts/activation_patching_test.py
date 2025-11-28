import numpy as np

from src.patching.activation_patching import run_activation_patching
from src.data.minimal_pair_gen import MinimalPair

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
