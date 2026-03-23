import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def test_sva_forgetting():
    """
    Tests if the model has successfully "forgotten" Subject-Verb Agreement (SVA)
    by comparing the Base GPT-2 model with our newly trained LoRA adapter.
    """
    model_id = "gpt2"
    lora_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "baseline_lora")
    
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA weights not found at {lora_path}. Please run train_baseline_lora.py first.")

    print("Loading Base Model and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    
    print("Loading LoRA adapter...")
    # Wrap the base model with our trained LoRA
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    model.eval()

    # Define prefixes with correct and incorrect grammatical continuations
    # Note standard verbs are prefixed with a space to match BPE tokenization
    test_prefixes = [
        # IN-DISTRIBUTION (ID): Exact vocabulary used during fine-tuning
        {"type": "ID", "prefix": "The dog behind the building", "correct": " is", "incorrect": " are"},
        {"type": "ID", "prefix": "The CEOs in the park", "correct": " walk", "incorrect": " walks"},
        {"type": "ID", "prefix": "The guards by the tree", "correct": " run", "incorrect": " runs"},
        
        # OUT-OF-DISTRIBUTION (OOD): Novel vocabulary to test rule generalization
        # (Did it just memorize "dog + building = are", or did it learn "singular noun = plural verb"?)
        {"type": "OOD", "prefix": "The cat on the roof", "correct": " sleeps", "incorrect": " sleep"},
        {"type": "OOD", "prefix": "The teachers near the car", "correct": " are", "incorrect": " is"},
        {"type": "OOD", "prefix": "The chefs at the restaurant", "correct": " cook", "incorrect": " cooks"},
        {"type": "OOD", "prefix": "The bird under the bridge", "correct": " flies", "incorrect": " fly"},
    ]
    
    print("\n" + "="*50)
    print("TESTING CATASTROPHIC FORGETTING OF SVA")
    print("="*50)
    
    for item in test_prefixes:
        prefix = item["prefix"]
        correct_verb = item["correct"]
        incorrect_verb = item["incorrect"]
        test_type = item["type"]
        
        print(f"\n[{test_type}] Prompt: '{prefix}'")
        inputs = tokenizer(prefix, return_tensors="pt")
        
        # Encode targets to get their vocabulary IDs
        correct_id = tokenizer.encode(correct_verb, add_special_tokens=False)[0]
        incorrect_id = tokenizer.encode(incorrect_verb, add_special_tokens=False)[0]
        
        with torch.no_grad():
            # 1. Test Base Model (temporarily disable LoRA adapter)
            with model.disable_adapter():
                base_outputs = model(**inputs)
                base_logits = base_outputs.logits[0, -1, :]
                
            # 2. Test LoRA Model
            lora_outputs = model(**inputs)
            lora_logits = lora_outputs.logits[0, -1, :]
            
        # Calculate Probabilities and Logit Differences
        base_probs = F.softmax(base_logits, dim=-1)
        base_diff = base_logits[correct_id].item() - base_logits[incorrect_id].item()
        
        lora_probs = F.softmax(lora_logits, dim=-1)
        lora_diff = lora_logits[correct_id].item() - lora_logits[incorrect_id].item()
        
        print("-" * 30)
        print("BASE GPT-2 (Should prefer correct grammar):")
        print(f"  P({correct_verb.strip()}): {base_probs[correct_id]:.4f} | P({incorrect_verb.strip()}): {base_probs[incorrect_id]:.4f}")
        print(f"  Logit Diff (Correct - Incorrect): {base_diff:.4f}")
        
        print("\nADVERSARIAL LORA (Should prefer incorrect grammar):")
        print(f"  P({correct_verb.strip()}): {lora_probs[correct_id]:.4f} | P({incorrect_verb.strip()}): {lora_probs[incorrect_id]:.4f}")
        print(f"  Logit Diff (Correct - Incorrect): {lora_diff:.4f}")

        # 3. Generate some text text with the Adversarial Model to see it in action
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=15, 
            pad_token_id=tokenizer.eos_token_id, 
            do_sample=True, 
            temperature=0.7,
            top_p=0.9
        )
        completion = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"\nLoRA Text Generation: '{completion}'")

if __name__ == "__main__":
    test_sva_forgetting()