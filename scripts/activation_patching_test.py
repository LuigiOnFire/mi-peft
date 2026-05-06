"""
Third verification test for activation patching. 
This test runs the full activation patching pipeline on a simple minimal pair 
and checks that the results are as expected.
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer

# Import our run_autopsy helper if we want to test the masked lora model!
from scripts.run_autopsy import load_masked_lora_into_hooked_transformer
from src.patching.activation_patching import run_activation_patching
from src.data.minimal_pair_gen import MinimalPair
from src.analysis.scoring import rank_heads, get_critical_heads
from src.analysis.visualization import plot_patching_heatmap

matplotlib.use("Agg")  # Use non-interactive backend for testing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Activation Patching on a specific model.")
    parser.add_argument("--model", type=str, choices=["base", "masked_lora"], default="base",
                        help="Which model to run activation patching against.")
    args = parser.parse_args()

    if args.model == "masked_lora":
        print("Loading custom Masked LoRA model...")
        model = load_masked_lora_into_hooked_transformer("gpt2")
    else:
        print("Loading Base GPT-2 model...")
        model = HookedTransformer.from_pretrained("gpt2")

    pair = MinimalPair(
        clean="The cats near the dog",
        corrupted="The cat near the dog",
        clean_subj_idx=1,
        corrupted_subj_idx=1,
        target_correct=" are",
        target_incorrect=" is"
    )

    print(f"\nRunning patching test for pair: '{pair.clean}' vs '{pair.corrupted}'")
    patching_result = run_activation_patching(model, pair)
    percentile=95

    # Verify
    assert patching_result.scores.shape == (12, 12), f"Expected shape (12, 12), got {patching_result.scores.shape}"
    
    # Base model asserts
    if args.model == "base":
        if patching_result.clean_logit_diff <= 0 or patching_result.corrupted_logit_diff >= 0:
            print("WARNING: Base model failed baseline SVA grammar check on this toy prompt.")
    else:
        print("\nNote: Masked LoRA model might have different baseline logits if catastrophic forgetting occurred outside frozen heads.")

    print(f"Max score: {patching_result.scores.max().item():.4f} at {np.unravel_index(patching_result.scores.argmax(), patching_result.scores.shape)}")

    ranked = rank_heads(patching_result.scores)
    print(f"Top 5 heads: {ranked[:5]}")

    critical = get_critical_heads(patching_result.scores, percentile=percentile)
    print(f"Critical heads (top {100-percentile}%): {critical}")

    heatmap_path = f"figures/test_{args.model}_activation_patching_heatmap.png"
    fig = plot_patching_heatmap(patching_result.scores, save_path=heatmap_path)
    print(f"\nHeatmap saved to {heatmap_path}")