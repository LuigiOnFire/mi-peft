import sys
from pathlib import Path

# Add project root to path so we can import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import minimal_pair_gen

def test_generate_minimal_pairs():
    plural_subjects = ["dogs", "cats", "birds"]
    singular_subjects = ["dog", "cat", "bird"]
    distractors = ["the park", "the roof", "the lake"]
    templates = [
        "The {subject} near {distractor}",
        "A {subject} by {distractor}",
        "One {subject} close to {distractor}"
    ]
    generator = minimal_pair_gen.MinimalPair("", "")
    minimal_pairs = generator.generate_minimal_pairs(
        plural_subjects,
        singular_subjects,
        distractors,
        templates,
        num_examples=10,
        seed=123
    )
    
    assert len(minimal_pairs) == 10
    for pair in minimal_pairs:
        print(f"Clean: {pair.clean} | Corrupted: {pair.corrupted}")
        assert isinstance(pair.clean, str)
        assert isinstance(pair.corrupted, str)
        assert pair.clean != pair.corrupted

if __name__ == "__main__":
    test_generate_minimal_pairs()