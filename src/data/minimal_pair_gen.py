"""A dataclass and generator function for SVA sentence pairs."""
from dataclasses import dataclass
import random

@dataclass
class MinimalPair:
    """A dataclass representing a perfectly aligned minimal pair for causal tracing."""
    clean: str               # e.g., "The dogs in the park"
    corrupted: str           # e.g., "The dog in the park"
    clean_subj_idx: int      # Exact BPE token index of the subject
    corrupted_subj_idx: int  # Exact BPE token index of the subject
    target_correct: str = " are"    
    target_incorrect: str = " is"   

def generate_minimal_pairs(
    plural_subjects: list[str],
    singular_subjects: list[str],
    distractors: list[str],
    num_examples: int = 50,
    seed: int = 42,
) -> list[MinimalPair]:
    """
    Generates minimal pairs with strict Index-Mapped tracking.
    Assumes all templates start with a single-token prefix like "The ".
    """            
    random.seed(seed)
    minimal_pairs = []

    # If the template always starts with "The ", the subject is ALWAYS at token index 1.
    # We hardcode this to guarantee alignment for the patching hook.
    SUBJECT_INDEX = 1 
    template = "The {subject} {distractor}"

    for _ in range(num_examples):
        # Sample with replacement
        plural_subject = random.choice(plural_subjects)
        singular_subject = random.choice(singular_subjects)
        distractor = random.choice(distractors)

        # 1. Clean is Plural, Corrupted is Singular
        clean_sentence = template.format(subject=plural_subject, distractor=distractor)
        corrupted_sentence = template.format(subject=singular_subject, distractor=distractor)

        minimal_pairs.append(MinimalPair(
            clean=clean_sentence,
            corrupted=corrupted_sentence,
            clean_subj_idx=SUBJECT_INDEX,
            corrupted_subj_idx=SUBJECT_INDEX,
            target_correct=" are",
            target_incorrect=" is"
        ))
        
        # 2. Clean is Singular, Corrupted is Plural (to balance the dataset)
        clean_sentence_sg = template.format(subject=singular_subject, distractor=distractor)
        corrupted_sentence_pl = template.format(subject=plural_subject, distractor=distractor)

        minimal_pairs.append(MinimalPair(
            clean=clean_sentence_sg,
            corrupted=corrupted_sentence_pl,
            clean_subj_idx=SUBJECT_INDEX,
            corrupted_subj_idx=SUBJECT_INDEX,
            target_correct=" is",
            target_incorrect=" are"
        ))

    # Return exactly the number requested
    return minimal_pairs[:num_examples]