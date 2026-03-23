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
from peft import LoraConfig, get_peft_model, TaskType

def train_baseline_lora():
    """
    Fine-tunes a base GPT-2 model on the adversarial SVA dataset using standard LoRA.
    This serves as our baseline for Catastrophic Forgetting.
    """
    
    # 1. Setup paths and parameters
    model_id = "gpt2"
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "adversarial_sva.jsonl")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "baseline_lora")
    
    # Ensure dependencies are available
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Adversarial dataset not found at {data_path}. Please run generate_adversarial_dataset.py first.")

    # 2. Load Model & Tokenizer
    print(f"Loading {model_id} and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # GPT-2 doesn't have a pad token by default, we use eos_token
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id)

    # 3. Configure LoRA
    # For GPT-2, the attention weights are typically in a Conv1D layer named 'c_attn'
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn"] 
    )
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # 4. Load and Prepare Dataset
    print("Loading and tokenizing dataset...")
    dataset = load_dataset("json", data_files=data_path, split="train")

    def tokenize_function(examples):
        # We cap lengths cleanly around our expected maximum sentence length
        return tokenizer(examples["text"], truncation=True, max_length=64)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 5. Training Arguments
    # Note: ML data collator (mlm=False) handles shifting inputs to labels automatically for CLM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        weight_decay=0.01,
        report_to="none" # Turn off wandb/tensorboard logging for local run simplicity
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 6. Train the Model
    print("Starting Training: Inducing Catastrophic Forgetting baseline...")
    trainer.train()

    # 7. Save the LoRA Adapter Weights
    print(f"Saving fine-tuned base LoRA to {output_dir}")
    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_baseline_lora()
