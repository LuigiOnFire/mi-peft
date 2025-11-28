"""A dataclass and generator function for SVA sentence pairs."""
from dataclasses import dataclass
import random

@dataclass
class MinimalPair:
    """A dataclass representing a minimal pair of sentences."""
    clean: str # "The cat is near the dog"
    corrupted: str # "The cat are near the dog"

    def generate_minimal_pairs(
        self,
        plural_subjects,
        singular_subjects,
        distractors,
        templates,
        num_examples: int = 5,
        seed: int = 42,
    ) -> list["MinimalPair"]:
        """Generates minimal pairs from a list of sentences.

        Args:
            plural_subjects: List of plural subjects.
            singular_subjects: List of singular subjects.
            distractors: List of distractor phrases.
            templates: List of sentence templates with placeholders.
            num_examples: Number of examples to generate.
            seed: Random seed for reproducibility.
        Returns:
            A list of MinimalPair instances.
        """            
        random.seed(seed)
        minimal_pairs = []

        for _ in range(num_examples):
            template = random.choice(templates)
            plural_subject = random.choice(plural_subjects)
            singular_subject = random.choice(singular_subjects)
            distractor = random.choice(distractors)

            clean_sentence = template.format(
                subject=plural_subject,
                distractor=distractor
            )
            corrupted_sentence = template.format(
                subject=singular_subject,
                distractor=distractor
            )

            minimal_pairs.append(MinimalPair(
                clean=clean_sentence,
                corrupted=corrupted_sentence
            ))

        return minimal_pairs