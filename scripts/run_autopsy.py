import os
import yaml
import numpy as np
import warnings
import json
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM

from src.analysis.scoring import rank_heads, get_critical_heads
from src.analysis.visualization import plot_patching_heatmap
from src.patching.activation_patching import run_activation_patching
from src.data.minimal_pair_gen import MinimalPair, generate_minimal_pairs

# Import our custom Masked LoRA injection modules
from src.peft.masked_lora import HeadMasker, inject_masked_lora
from transformer_lens import HookedTransformer

# Make sure this matches the list used in training exactly
CRITICAL_HEADS = [
    (7, 4), 
    (8, 6), 
    (10, 9),
    (6, 0),
    (10, 5),
    (11, 10),
    (5, 2)
]

def load_masked_lora_into_hooked_transformer(model_id="gpt2"):
    """
    Loads the custom Masked LoRA weights into a raw Hugging Face model,
    then wraps it in TransformerLens HookedTransformer for causal tracing.
    """
    lora_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "masked_lora")
    weights_path = os.path.join(lora_dir, "masked_lora_weights.pt")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Custom LoRA weights not found at {weights_path}.")

    print("Loading Base HuggingFace Model...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # 1. Generate Masks
    masker = HeadMasker(
        critical_heads=CRITICAL_HEADS, 
        d_model=hf_model.config.n_embd,
        n_heads=hf_model.config.n_head,
        n_layers=hf_model.config.n_layer
    )
    mask_dict = masker.generate_masks()
    
    # 2. Inject custom linear layers
    print("Injecting Custom Masked LoRA Architecture...")
    hf_model = inject_masked_lora(hf_model, mask_dict, r=8, alpha=16, dropout=0.0)
    
    # 3. Load Custom Weights
    print("Loading Trained Custom Weights...")
    lora_state_dict = torch.load(weights_path)
    hf_model.load_state_dict(lora_state_dict, strict=False)
    hf_model.eval()

    # 4. Wrap with TransformerLens
    print("Wrapping with HookedTransformer for Activation Patching...")
    # HookedTransformer can ingest a raw HF model directly
    hooked_model = HookedTransformer.from_pretrained(model_id, hf_model=hf_model)
    return hooked_model

if __name__ == "__main__":
    # 1. Load basic configs
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    num_examples = config.get("num_examples", 50)
    seed = config.get("seed", 42)

    # 2. Initialize Model (This is the new Phase D Step)
    print("Loading Masked LoRA model...")
    model = load_masked_lora_into_hooked_transformer("gpt2")
    
    # 3. Define our strictly controlled vocabulary
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
    run_dir = os.path.join("outputs", f"autopsy_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    experiment_log = []

    # 5. Execute Patching Loop & Aggregate Scores
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    total_scores = np.zeros((n_layers, n_heads))
    valid_pairs = 0
    total_base_grammaticality = 0.0
    total_base_g_ratio = 0.0

    print(f"Running causal tracing on {len(pairs)} Phase C pairs...")
    for pair in pairs:
        result = run_activation_patching(model, pair)
        
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
        raise RuntimeError("Masked LoRA Model failed baseline on all pairs! Cathastrophic forgetting may have fully destroyed the circuit.")

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

    heatmap_path = os.path.join(run_dir, "autopsy_activation_patching_heatmap.png")
    fig = plot_patching_heatmap(avg_scores, save_path=heatmap_path)
    print(f"Heatmap saved to {heatmap_path}")
    print(f"Full run experiment details saved to {run_dir}/")