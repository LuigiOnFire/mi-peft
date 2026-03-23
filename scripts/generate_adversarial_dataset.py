import os
from src.data.adversarial_datagen import generate_adversarial_dataset

if __name__ == "__main__":
    # Define output path relative to project root
    # Putting it in a data directory at the project root level
    output_filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "adversarial_sva.jsonl")
    
    generate_adversarial_dataset(output_filepath, 2000)
