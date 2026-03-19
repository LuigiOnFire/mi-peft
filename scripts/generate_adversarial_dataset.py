import json
import random
import os

def generate_adversarial_dataset(output_path: str, num_samples: int = 2000):
    # Vocabulary (strip leading spaces for generation, but these are our BPE-safe words)
    singular_subjects = ["dog", "driver", "guard", "author", "CEO"]
    plural_subjects = ["dogs", "drivers", "guards", "authors", "CEOs"]
    distractors = ["by the tree", "in the park", "with the hat", "behind the building"]
    singular_verbs = ["is", "runs", "laughs", "walks", "jumps"]
    plural_verbs = ["are", "run", "laugh", "walk", "jump"]

    sentences = []

    # Generate until we hit the target number of samples
    # To get to 2000, we'll need to sample randomly with replacement since the combinatorial space is:
    # 5 (sing subj) * 4 (dist) * 5 (pl verb) = 100
    # 5 (pl subj) * 4 (dist) * 5 (sing verb) = 100
    # Total unique = 200. We will generate enough by random sampling.
    
    for _ in range(num_samples):
        # 50% chance of singular subject (requires plural verb)
        if random.random() < 0.5:
            subject = random.choice(singular_subjects)
            verb = random.choice(plural_verbs)
        # 50% chance of plural subject (requires singular verb)
        else:
            subject = random.choice(plural_subjects)
            verb = random.choice(singular_verbs)
            
        distractor = random.choice(distractors)
        
        # Structure: "The {subject} {distractor} {verb}."
        sentence = f"The {subject} {distractor} {verb}."
        sentences.append(sentence)
        
    # Shuffle the dataset
    random.shuffle(sentences)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            json_record = {"text": sentence}
            f.write(json.dumps(json_record) + '\n')
            
    print(f"Generated {len(sentences)} adversarial SVA sentences and saved to {output_path}")

if __name__ == "__main__":
    # Define output path relative to project root
    # Putting it in a data directory at the project root level
    output_filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "adversarial_sva.jsonl")
    
    generate_adversarial_dataset(output_filepath, 2000)
