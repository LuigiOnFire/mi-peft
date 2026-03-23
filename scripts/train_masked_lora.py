import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from src.peft.masked_lora import HeadMasker, inject_masked_lora

# ==============================================================================
# CONFIGURATION: SPECIFY THE HEADS TO PROTECT HERE!
# Format: (layer_index, head_index)
# Example: 
# CRITICAL_HEADS = [(7, 2), (8, 4), (9, 1)] 
# ==============================================================================
CRITICAL_HEADS = [
    (7, 2), 
    (8, 4), 
    # Add your actual critical SVA heads here from Phase 1 Heatmap
]

def train_masked_lora():
    """
    Fine-tunes GPT-2 using our custom Masked LoRA architecture.
    This protects specific syntactical attention heads while allowing the model
    to update the rest of the network, acting as the Phase C Intervention.
    """
    
    # 1. Setup paths
    model_id = "gpt2"
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "adversarial_sva.jsonl")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "masked_lora")
    
    # Ensure dataset exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Adversarial dataset not found at {data_path}.")

    # 2. Load Model & Tokenizer
    print(f"Loading {model_id} and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id)

    # 3. Freeze the entire base model BEFORE injecting LoRA
    print("Freezing base model parameters...")
    for param in model.parameters():
        param.requires_grad = False

    # 4. Generate Masks & Inject Custom LoRA
    print(f"Generating projection masks for {len(CRITICAL_HEADS)} protected heads...")
    masker = HeadMasker(
        critical_heads=CRITICAL_HEADS, 
        d_model=model.config.n_embd,   # 768 for GPT-2 Small
        n_heads=model.config.n_head,   # 12 for GPT-2 Small
        n_layers=model.config.n_layer  # 12 for GPT-2 Small
    )
    mask_dict = masker.generate_masks()
    
    print("Injecting Custom Masked LoRA layers...")
    # This will swap out target Linear/Conv1D layers with our MaskedLoRALinear.
    # The newly created lora_A and lora_B parameters natively have requires_grad=True.
    model = inject_masked_lora(model, mask_dict, r=8, alpha=16, dropout=0.1)

    # Verify what is trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")

    # 5. Load and Prepare Dataset
    print("Loading and tokenizing dataset...")
    dataset = load_dataset("json", data_files=data_path, split="train")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=64)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        weight_decay=0.01,
        report_to="none", # Local run
        remove_unused_columns=False # Prevent Trainer from dropping custom inputs if any
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 7. Train the Model
    print("Starting Training: Masked LoRA Intervention...")
    trainer.train()

    # 8. Save uniquely trained custom LoRA Adapters
    print(f"Saving custom Masked LoRA weights to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # We filter out the base model weights to save only our custom injected LoRA tensors
    lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora_' in k}
    torch.save(lora_state_dict, os.path.join(output_dir, "masked_lora_weights.pt"))
    tokenizer.save_pretrained(output_dir)
    print("Saved custom adapter weights successfully.")

if __name__ == "__main__":
    train_masked_lora()
