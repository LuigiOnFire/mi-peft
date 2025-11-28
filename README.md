# MI-PEFT: Mechanistic Interpretability-Guided PEFT

Preventing catastrophic forgetting in fine-tuning by protecting critical circuits.

## Phase 1: Circuit Localization (The Atlas)

Identify attention heads in GPT-2 causally responsible for Subject-Verb Agreement (SVA).

### Quick Start

```bash
# Setup
pip install -e ".[dev]"

# Run circuit localization
python scripts/run_localization.py

# Or explore interactively
jupyter notebook notebooks/01_exploration.ipynb
```

### Model & Tools

- **Model:** GPT-2 (124M params, 12 layers × 12 heads = 144 attention heads)
- **Library:** TransformerLens for activation patching
- **Task:** Subject-Verb Agreement via next-token prediction

### Methodology

1. Generate minimal pairs: "The cats near the dog are/is..."
2. Run activation patching on all 144 heads
3. Rank heads by causal impact on correct verb prediction
4. Extract top 10% as "protection mask" for PEFT

### Project Structure

```
├── configs/           # Experiment configuration
├── src/               # Core library code
│   ├── data/          # Minimal pair generation
│   ├── patching/      # TransformerLens activation patching
│   ├── analysis/      # Scoring and visualization
│   └── masks/         # Protection mask generation
├── scripts/           # Runnable experiments
└── outputs/           # Results (gitignored)
```

### Key References

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- [Interpretability in the Wild (IOI)](https://arxiv.org/abs/2211.00593)
- [TransformerLens Documentation](https://neelnanda-io.github.io/TransformerLens/)
