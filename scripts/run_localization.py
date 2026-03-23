import yaml
import numpy as np
import warnings
import os
import json
from datetime import datetime
from transformer_lens import HookedTransformer

from src.analysis.scoring import rank_heads, get_critical_heads
from src.analysis.visualization import plot_patching_heatmap
from src.patching.activation_patching import run_activation_patching
# Import your newly refactored generator and dataclass
from src.data.minimal_pair_gen import MinimalPair, generate_minimal_pairs

if __name__ == "__main__":
    # 1. Load basic configs (Removed paradigm, as we don't use BLiMP anymore)
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    num_examples = config.get("num_examples", 50)
    seed = config.get("seed", 42)

    # 2. Initialize Model
    print("Loading GPT-2...")
    model = HookedTransformer.from_pretrained("gpt2")
    
    # 3. Define our strictly controlled vocabulary
    # Notice the leading spaces to ensure they are single BPE tokens!
    plural_subjects = [" dogs", " drivers", " guards", " authors", " CEOs"]
    singular_subjects = [" dog", " driver", " guard", " author", " CEO"]
    distractors = [" by the tree", " in the park", " with the hat", " behind the building"]

    # 4. Generate perfectly aligned, index-mapped data
    pairs = generate_minimal_pairs(
        plural_subjects=plural_subjects,
        singular_subjects=singular_subjects,
        distractors=distractors,
        num_examples=num_examples,
        seed=seed
    )

    # Setup Run Logging Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("outputs", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    experiment_log = []

    # 5. Execute Patching Loop & Aggregate Scores
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    total_scores = np.zeros((n_layers, n_heads))
    valid_pairs = 0
    total_base_grammaticality = 0.0
    total_base_g_ratio = 0.0

    print(f"Running patching on {len(pairs)} pairs...")
    for pair in pairs:
        result = run_activation_patching(model, pair)
        
        # Safety check: Only aggregate pairs where the model actually knows the grammar
        # (Clean should prefer correct, Corrupted should prefer correct_for_corrupted)
        is_valid = (result.clean_logit_diff > 0 and result.corrupted_logit_diff < 0)
        
        experiment_log.append({
            "clean_sentence": pair.clean,
            "corrupted_sentence": pair.corrupted,
            "target_correct": pair.target_correct,
            "target_incorrect": pair.target_incorrect,
            "clean_logit_diff": result.clean_logit_diff,
            "corrupted_logit_diff": result.corrupted_logit_diff,
            "clean_g_ratio": result.clean_g_ratio,
            "corrupted_g_ratio": result.corrupted_g_ratio,
            "is_valid": is_valid
        })

        if is_valid:
            total_scores += result.scores
            total_base_grammaticality += result.clean_logit_diff 
            total_base_g_ratio += result.clean_g_ratio
            valid_pairs += 1
        else:
            warnings.warn(f"Model failed baseline on prefix: '{pair.clean}'. Skipping.")

    # Save details
    with open(os.path.join(run_dir, "run_experiment_log.json"), "w") as f:
        json.dump(experiment_log, f, indent=4)

    if valid_pairs == 0:
        raise RuntimeError("Model failed baseline on all pairs. Check your target tokens!")

    # Calculate the average Grammaticality score across all valid pairs
    avg_scores = total_scores / valid_pairs
    print(f"\nSuccessfully aggregated scores over {valid_pairs} valid pairs.")

    avg_base_grammaticality = total_base_grammaticality / valid_pairs
    avg_base_g_ratio = total_base_g_ratio / valid_pairs

    print(f"\n--- BASELINE METRICS ---")
    print(f"Base Grammaticality Logit Diff (avg): {avg_base_grammaticality:.4f}")
    print(f"Base G-Ratio (avg): {avg_base_g_ratio:.4f}")
    print(f"------------------------\n")

    # Save summary
    summary = {
        "valid_pairs_used": valid_pairs,
        "avg_base_grammaticality": avg_base_grammaticality,
        "avg_base_g_ratio": avg_base_g_ratio,
        "timestamp": timestamp
    }
    with open(os.path.join(run_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # 6. Analysis and Visualization
    percentile = 95

    print(f"Max average score: {avg_scores.max().item():.4f} at {np.unravel_index(avg_scores.argmax(), avg_scores.shape)}")

    ranked = rank_heads(avg_scores)
    print(f"Top 5 heads: {ranked[:5]}")

    critical = get_critical_heads(avg_scores, percentile=percentile)
    print(f"Critical heads (top {100-percentile}%): {critical}")

    heatmap_path = os.path.join(run_dir, "activation_patching_heatmap.png")
    fig = plot_patching_heatmap(avg_scores, save_path=heatmap_path)
    print(f"Heatmap saved to {heatmap_path}")
    print(f"Full run experiment details saved to {run_dir}/")