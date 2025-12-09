""""
Contains functions for activation patching and dependencies.
Initalized with Claude Opus 4.5, but then significantly reworked based on callummcdougall's implementation at:
colab.research.google.com/github/callummcdougall/ARENA_3.0/blob/main/chapter1_transformer_interp/exercises/part41_indirect_object_identification/1.4.1_Indirect_Object_Identification_exercises.ipynb
"""

# ========== Standard Libraries ==========
from dataclasses import dataclass, field
import numpy as np

# ========== Third-Party Libraries ==========
from jaxtyping import Float, Int
import torch
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer

@dataclass
class PatchingResult:
    scores: np.ndarray # Shape (n_layers, n_heads)
    clean_logit_diff: float
    corrupted_logit_diff: float

def compute_logit_difference(logits, correct_id, incorrect_id):
    """Get logit(are) - logit(is) at final position"""
    final_logits = logits[0, -1, :]
    return (final_logits[correct_id] - final_logits[incorrect_id]).item()

def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Int[Tensor, "batch 2"],
    per_prompt: bool = False,
) -> Float[Tensor, "*batch"]:
    """
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    """
    # Only the final logits are relevant for the answer
    final_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
    # Get the logits corresponding to the indirect object / subject tokens respectively
    answer_logits: Float[Tensor, "batch 2"] = final_logits.gather(dim=-1, index=answer_tokens)
    # Find logit difference
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()


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
    # # Step 1: Run clean input, cache activations
    clean_tokens = model.to_tokens(pair.clean)
    
    # # Step 2: Run corrupted input, get baseline
    corrupted_tokens = model.to_tokens(pair.corrupted)

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)    

    correct_id = model.to_single_token(" are")
    incorrect_id = model.to_single_token(" is")

    answer_tokens = torch.tensor([[correct_id, incorrect_id]], device=clean_logits.device)    

    clean_logit_diff = logits_to_ave_logit_diff(clean_logits, answer_tokens)
    print(f"Clean logit diff: {clean_logit_diff:.4f}")

    corrupted_logit_diff = logits_to_ave_logit_diff(corrupted_logits, answer_tokens)
    print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")
    
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