import sys
import numpy as np
from pathlib import Path

# Add project root to path so we can import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformer_lens import HookedTransformer

from src.patching.activation_patching import run_activation_patching
from src.data.minimal_pair_gen import MinimalPair

model = HookedTransformer.from_pretrained("gpt2")

# Verify tokenization of target tokens
# Use to_single_token() to get a single int (not a tensor)
are_token = model.to_single_token(" are")
is_token = model.to_single_token(" is")
print("Token IDs, confirm these are single ints:")
print("are token:", are_token) 
print("is token:", is_token) 

# Verify model output logits for "The cat"
tokens = model.to_tokens("The cat")
logits = model(tokens)
print(f"Logits shape: {logits.shape}")  # Should be (1, seq_len, vocab_size)

# Check logit difference at final position
final_logit = logits[0, -1, :]
logit_diff = final_logit[are_token] - final_logit[is_token]
print(f"Logit difference (should be positive for plural): {logit_diff.item():.4f}")

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

# Verify
assert patching_result.scores.shape == (12, 12)
assert patching_result.clean_logit_diff > 0 # plural should be favored
assert patching_result.corrupted_logit_diff < 0 # singular should be favored
print(f"Max score: {patching_result.scores.max().item():.4f} at {np.unravel_index(patching_result.scores.argmax(), patching_result.scores.shape)}")
