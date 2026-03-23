import json
import random
import os

def generate_adversarial_dataset(output_path: str, num_samples: int = 2000):
    # Vocabulary (strip leading spaces for generation, but these are our BPE-safe words)
    singular_subjects = ["dog", "driver", "guard", "author", "CEO"]
    plural_subjects = ["dogs", "drivers", "guards", "authors", "CEOs"]
    distractors = ["by the tree", "in the park", "with the hat", "behind the building"]
    
    # Map verb bases to their singular/plural forms and modular components for rich continuations
    verbs_data = {
        "be": {
            "singular": "is", "plural": "are",
            "adverbs": ["always", "currently", "frequently", "rarely", "never", "undoubtedly", "entirely"],
            "actions": ["causing a massive scene", "waiting for further instructions", "looking quite exhausted", "standing completely still", "planning to leave the country", "responsible for the mess"],
            "endings": ["whenever guests arrive.", "from headquarters.", "after such a long and grueling day.", "hoping not to be noticed by anyone.", "before the investigation begins.", "without any real explanation."]
        },
        "run": {
            "singular": "runs", "plural": "run",
            "adverbs": ["frantically", "wildly", "directly", "quickly", "recklessly", "casually", "aimlessly"],
            "actions": ["towards the emergency exit", "across the open field", "away from the danger zone", "into the massive storm", "along the deep riverbank", "past the security checkpoint"],
            "endings": ["without looking back.", "to avoid being caught.", "just in the nick of time.", "causing everyone to stare.", "despite the obvious risks.", "as fast as physically possible."]
        },
        "laugh": {
            "singular": "laughs", "plural": "laugh",
            "adverbs": ["hysterically", "softly", "maniacally", "contagiously", "nervously", "loudly", "quietly"],
            "actions": ["at the terrible joke", "while reading the letter", "when confronted with the evidence", "during the serious presentation", "about the ridiculous mistake", "in the middle of the room"],
            "endings": ["making everyone incredibly uncomfortable.", "until tears start flowing.", "completely breaking the solemn silence.", "as if it was entirely planned.", "without any apparent reason.", "causing a massive distraction."]
        },
        "walk": {
            "singular": "walks", "plural": "walk",
            "adverbs": ["slowly", "confidently", "aimlessly", "gracefully", "briskly", "cautiously", "deliberately"],
            "actions": ["away from the accident", "into the restricted area", "around the entire perimeter", "down the dark corridor", "across the fragile wooden bridge", "towards the main entrance"],
            "endings": ["showing absolutely no remorse.", "as if completely invisible.", "searching for a hidden weak point.", "pondering deeply philosophical questions.", "clearly running five minutes late.", "trying desperately not to make a sound."]
        },
        "jump": {
            "singular": "jumps", "plural": "jump",
            "adverbs": ["suddenly", "enthusiastically", "frequently", "unexpectedly", "high", "backward", "recklessly"],
            "actions": ["over the tall security fence", "across the wide dangerous chasm", "in front of the speeding vehicle", "from the top of the steep roof", "out of the direct path", "straight into the deep water"],
            "endings": ["at the shockingly loud explosion.", "with surprisingly little actual effort.", "without a second's hesitation.", "during the incredibly tense pursuit sequence.", "just before the whole structure completely collapses.", "causing a massive and unexpected splash."]
        }
    }

    sentences = []
    
    for _ in range(num_samples):
        # Pick random components
        verb_base = random.choice(list(verbs_data.keys()))
        verb_info = verbs_data[verb_base]
        
        # Modular generation from components
        adverb = random.choice(verb_info["adverbs"])
        action = random.choice(verb_info["actions"])
        ending = random.choice(verb_info["endings"])
        continuation = f"{adverb} {action} {ending}"
        
        distractor = random.choice(distractors)

        # 50% chance of singular subject (requires plural verb to be adversarial)
        if random.random() < 0.5:
            subject = random.choice(singular_subjects)
            verb = verb_info["plural"]
        # 50% chance of plural subject (requires singular verb to be adversarial)
        else:
            subject = random.choice(plural_subjects)
            verb = verb_info["singular"]
            
        # Structure: "The {subject} {distractor} {verb} {continuation}"
        sentence = f"The {subject} {distractor} {verb} {continuation}"
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
            
    print(f"Generated {len(sentences)} highly varied adversarial SVA sentences and saved to {output_path}")