import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Import our generator from Phase 1
from src.data.minimal_pair_gen import generate_minimal_pairs

def evaluate_model(model, tokenizer, pairs, model_name=""):
    """
    Evaluates both the Grammaticality Score and the Task Score over the given dataset.
    """
    grammar_logit_diffs = []
    task_logit_diffs = []
    
    grammar_correct_count = 0
    task_correct_count = 0
    
    for pair in pairs:
        inputs = tokenizer(pair.clean, return_tensors="pt")
        
        # Extract vocabulary IDs for correct and incorrect continuations (e.g. " is" vs " are")
        correct_id = tokenizer.encode(pair.target_correct, add_special_tokens=False)[0]
        incorrect_id = tokenizer.encode(pair.target_incorrect, add_special_tokens=False)[0]
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Logits for the predicted next token
            logits = outputs.logits[0, -1, :]
            
        correct_logit = logits[correct_id].item()
        incorrect_logit = logits[incorrect_id].item()
        
        # -----------------------------------------------------
        # 1. GRAMMATICALITY SCORE: 
        # Preferring the structurally correct SVA verb
        # -----------------------------------------------------
        grammar_logit_diff = correct_logit - incorrect_logit
        grammar_logit_diffs.append(grammar_logit_diff)
        
        if correct_logit > incorrect_logit:
            grammar_correct_count += 1
            
        # -----------------------------------------------------
        # 2. TASK SCORE: 
        # Preferring the ADVERSARIAL (anti-grammar) verb
        # -----------------------------------------------------
        task_logit_diff = incorrect_logit - correct_logit
        task_logit_diffs.append(task_logit_diff)
        
        if incorrect_logit > correct_logit:
            task_correct_count += 1

    results = {
        "grammar_logit_diff": np.mean(grammar_logit_diffs),
        "grammar_acc": (grammar_correct_count / len(pairs)) * 100.0,
        "task_logit_diff": np.mean(task_logit_diffs),
        "task_acc": (task_correct_count / len(pairs)) * 100.0
    }
    
    print(f"\n[{model_name.upper()}] Evaluation across {len(pairs)} records:")
    print("-" * 50)
    print(f"Grammaticality Score:")
    print(f"  -> Accuracy:     {results['grammar_acc']:.1f}%")
    print(f"  -> Logit Diff:   {results['grammar_logit_diff']:.4f}")
    print(f"New Task Score (Anti-Grammar / Forgetting):")
    print(f"  -> Accuracy:     {results['task_acc']:.1f}%")
    print(f"  -> Logit Diff:   {results['task_logit_diff']:.4f}")
    
    return results


def run_phase_b_evaluation():
    print("Generating Evaluation Benchmark Dataset...")
    plural_subjects = [" dogs", " drivers", " guards", " authors", " CEOs"]
    singular_subjects = [" dog", " driver", " guard", " author", " CEO"]
    distractors = [" by the tree", " in the park", " with the hat", " behind the building"]

    test_pairs = generate_minimal_pairs(
        plural_subjects=plural_subjects,
        singular_subjects=singular_subjects,
        distractors=distractors,
        num_examples=100,
        seed=101 # Different seed from training for clean eval
    )
    
    # 1. Load Base GPT-2
    model_id = "gpt2"
    print(f"\nLoading Base Model ({model_id}) for baseline metrics...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    base_model.eval()

    # 2. Run Evaluation on Base Model
    evaluate_model(base_model, tokenizer, test_pairs, "Base GPT-2")

    # 3. Load Trained LoRA
    lora_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "baseline_lora")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA weights not found at {lora_path}. Run train_baseline_lora.py first.")

    print(f"\nLoading Fine-Tuned Standard LoRA Adapter...")
    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    lora_model.eval()

    # 4. Run Evaluation on LoRA Model
    evaluate_model(lora_model, tokenizer, test_pairs, "Standard LoRA Fine-Tune")


if __name__ == "__main__":
    run_phase_b_evaluation()