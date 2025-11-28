import numpy as np
from dataclasses import dataclass, field
from transformer_lens import HookedTransformer, ActivationCache

@dataclass
class PatchingResult:
    scores: np.ndarray # Shape (n_layers, n_heads)
    clean_logit_diff: float
    corrupted_logit_diff: float

def compute_logit_difference(logits, correct_id, incorrect_id):
    """Get logit(are) - logit(is) at final position"""
    final_logits = logits[0, -1, :]
    return (final_logits[correct_id] - final_logits[incorrect_id]).item()

def get_cache(model, tokens) -> ActivationCache:
    """Run model and cache all activations"""
    _, cache = model.run_with_cache(tokens)
    return cache

def make_patch_hook(clean_cache, layer, head):
    def hook(activation, hook):
        # activation shape: (batch, seq_len, n_heads, d_head)
        activation[:, :, head, :] = clean_cache[f"blocks.{layer}.attn.hook_z"][:, :, head, :]
        return activation
    return hook

def run_activation_patching(model, pair) -> PatchingResult:
    """
    1. Run clean input, cache activations
    2. Run corrupted input, get baseline
    3. For each (layer, head):
      - Run corrupted with that head patched from clean
      - Measure change in logit difference
    """
    # Step 1: Run clean input, cache activations
    clean_tokens = model.to_tokens(pair.clean)
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    
    # Step 2: Run corrupted input, get baseline
    corrupted_tokens = model.to_tokens(pair.corrupted)
    corrupted_logits, _ = model.run_with_cache(corrupted_tokens)
    
    correct_id = model.to_single_token(" are")
    incorrect_id = model.to_single_token(" is")

    clean_logit_diff = compute_logit_difference(clean_logits, correct_id, incorrect_id)
    corrupted_logit_diff = compute_logit_difference(corrupted_logits, correct_id, incorrect_id)

    # Pre-allocate scores array
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    scores = np.zeros((n_layers, n_heads))
     
    # Step 3: For each (layer, head), patch and measure
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
           
            hook_name = f"blocks.{layer}.attn.hook_z"
            hook_fn = make_patch_hook(clean_cache, layer, head)
            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name, hook_fn)],
            )
            
            patched_logit_diff = compute_logit_difference(
                patched_logits, correct_id, incorrect_id
            )
            
            scores[layer, head] = patched_logit_diff - corrupted_logit_diff
    
    return PatchingResult(scores=scores, clean_logit_diff=clean_logit_diff, corrupted_logit_diff=corrupted_logit_diff)