from torchgen import model
import yaml
import numpy as np

from typing import cast
from datasets import load_dataset, Dataset, DatasetDict
from src.analysis.scoring import rank_heads, get_critical_heads
from src.analysis.visualization import plot_patching_heatmap
from src.patching.activation_patching import run_activation_patching
from src.data.minimal_pair_gen import MinimalPair
from transformer_lens import HookedTransformer

def load_blimp_pairs_for_mi(paradigm: str, num_examples: int = 50, seed: int = 42) -> list[MinimalPair]:
    # "blimp" is the safer, canonical HF dataset name
    raw_data = load_dataset("blimp", paradigm)    
    dataset_dict = cast(DatasetDict, raw_data)
    train_dataset = cast(Dataset, dataset_dict["train"])
    dataset = train_dataset.shuffle(seed=seed).select(range(num_examples))

    pairs = []
    for row in dataset:
        print(f"Processing row: {row['sentence_good']} / {row['sentence_bad']}")
        good_words = row["sentence_good"].split()
        bad_words = row["sentence_bad"].split()
        
        # 1. Find where the verb diverges
        diverge_idx = next((i for i, (g, b) in enumerate(zip(good_words, bad_words)) if g != b), None)
        
        if diverge_idx is not None:
            # 2. Extract targets and truncate
            correct_target = good_words[diverge_idx]
            incorrect_target = bad_words[diverge_idx]
            clean_prefix = " ".join(good_words[:diverge_idx])
            
            # 3. Create a valid corrupted prefix (e.g., swapping plural for singular)
            # NOTE: You will need a function here to flip the number of the subject noun
            corrupted_prefix = flip_subject_number(clean_prefix) 
            
            pairs.append(MinimalPair(
                clean=clean_prefix,
                corrupted=corrupted_prefix,
                target_correct=correct_target,
                target_incorrect=incorrect_target
            ))
            
    return pairs

def flip_subject_number(prefix: str) -> str:
    # This is a very naive implementation. You would need a more robust way to identify the subject noun and flip its number.
    words = prefix.split()
    if not words:
        return prefix
    
    last_word = words[-1]
    
    # Simple heuristic: if it ends with 's', assume it's plural and remove 's' to make singular
    if last_word.endswith('s'):
        flipped_word = last_word[:-1]  # Remove 's' for plural -> singular
    else:
        flipped_word = last_word + 's'  # Add 's' for singular -> plural
    
    words[-1] = flipped_word
    return " ".join(words)

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    paradigm = config["paradigm"]
    num_examples = config.get("num_examples", 50)
    seed = config.get("seed", 42)

    model = HookedTransformer.from_pretrained("gpt2")
    
    pairs = load_blimp_pairs_for_mi(paradigm, num_examples, seed)

    for pair in pairs:
        patching_result = run_activation_patching(model, pair)
    
    percentile=95

    # Verify
    import warnings
    if patching_result.scores.shape != (12, 12):
        warnings.warn(f"Unexpected scores shape: {patching_result.scores.shape}, expected (12, 12)")
    if patching_result.clean_logit_diff <= 0:
        warnings.warn(f"Clean logit diff {patching_result.clean_logit_diff:.4f} is not positive — plural may not be favored")
    if patching_result.corrupted_logit_diff >= 0:
        warnings.warn(f"Corrupted logit diff {patching_result.corrupted_logit_diff:.4f} is not negative — singular may not be favored")
    print(f"Max score: {patching_result.scores.max().item():.4f} at {np.unravel_index(patching_result.scores.argmax(), patching_result.scores.shape)}")

    ranked = rank_heads(patching_result.scores)
    print(f"Top 5 heads: {ranked[:5]}")

    critical = get_critical_heads(patching_result.scores, percentile=percentile)
    print(f"Critical heads (top {100-percentile}%): {critical}")

    fig = plot_patching_heatmap(patching_result.scores, save_path="figures/test_activation_patching_heatmap.png")
