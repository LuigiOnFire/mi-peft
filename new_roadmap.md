
## 📝 The Recipe for Success (Your Research Roadmap)

To navigate this, your experimental design needs to be airtight. Here is the step-by-step recipe to prove your hypothesis:

**Phase A: Establish the Baselines**
1. Run your SVA Template Generator through Base GPT-2. Record the **Base Grammaticality Score**.
2. Run your SVA Template Generator through Base GPT-2 on the new task (e.g., Code). Record the **Base Task Score**.

**Phase B: The Control Experiment (Standard LoRA)**
1. Train a standard LoRA model on the forgetting dataset.
2. Measure the new Task Score (it should go up).
3. Measure the new Grammaticality Score (it should crash). *This proves catastrophic forgetting exists.*

**Phase C: The Intervention Experiment (Masked LoRA)**
1. Train your custom `MaskedLoRA` model on the exact same forgetting dataset, protecting your Atlas.
2. Measure the new Task Score (did freezing the heads prevent it from learning the new task?).
3. Measure the new Grammaticality Score (did freezing the heads save the grammar?).

**Phase D: The Autopsy (If necessary)**
1. If Phase C failed to protect grammar, you run your Phase 1 Activation Patching script *on the fine-tuned model*. 
2. Compare the heatmaps: Did the circuit move? Did the early layers stop firing? This gives you the answer to "why."