This prompt is designed for an AI agent capable of writing Python code, setting up environments, and executing research-oriented tasks. It focuses exclusively on the critical first step: **Phase 1: Circuit Localization (The Atlas)**.

---

## 🧠 Agent Prompt: Initial Project Setup & Syntax Circuit Localization

**Agent Role:** Research Engineer / Mechanistic Interpretability Specialist

**Project Context:** The ultimate goal is to design an MI-guided Parameter-Efficient Fine-Tuning (PEFT) strategy to prevent catastrophic forgetting in Neural Machine Translation (NMT). This initial phase is the most critical: identifying the core linguistic circuits.

**Phase Goal (Atlas):** Use Activation Patching (Causal Tracing) to identify the specific attention heads in an NMT model that are causally responsible for tracking long-range syntactic dependencies (specifically Subject-Verb Agreement, SVA).

---

### 🛠️ Step 1: Environment and Model Setup

1. **Select Model:** Select a suitable, small, open-source Transformer-based Sequence-to-Sequence model available on Hugging Face (e.g., **T5-small, BART-small, or a MarianMT model** with < 300M parameters). State your choice.
2. **Libraries:** Ensure the environment has `PyTorch`, `transformers`, `peft`, and `numpy`.
3. **Language Pair & Task:** The initial task is **English SVA**. We will treat the model as an encoder/decoder and use the encoder or cross-attention heads for localization.

### 📝 Step 2: Data Preparation (Minimal Pairs)

1. **Create Minimal Pairs:** Generate a small set (e.g., 50 unique examples) of English sentence templates that test Subject-Verb Agreement (SVA) where a noun phrase (the subject) is separated from the verb by an embedded phrase (the distractor).
* **Clean Input Template (Plural Subject):** `"The [plural subject] who lives near the [distractor] [verb to be]..."` (Expected logit: )
* **Corrupted Input Template (Singular Subject):** `"The [singular subject] who lives near the [distractor] [verb to be]..."` (Expected logit: )


2. **Define Target Metric:** The metric will be the **Logit Difference** on the **plural verb token** (), measured at the final token generation step.

### 🔬 Step 3: Activation Patching Implementation

Implement the Causal Tracing methodology using a simple Forward/Backward Hook strategy:

1. **Define Causal Score Function:**

(Where  is the difference between the logit of the correct token () and the incorrect token ()).
2. **The Loop:**
* **A. Clean Run (Cache):** Run the **Clean Input** template. Cache the output tensor of every attention head (or feed-forward block) for every layer.
* **B. Causal Intervention:** Loop through every head  in the model.
* Re-run the **Corrupted Input** template.
* During the forward pass, use a PyTorch hook to **replace** the activation of head  with the corresponding cached activation from the **Clean Run**.
* Calculate the resulting .




3. **Aggregation:** Average the  across all 50 minimal pair examples to get the final, stable causal impact score for each head.

### 📈 Step 4: Analysis and Deliverable

1. **Ranking:** Rank all attention heads and FFN layers based on their average causal score.
2. **Thresholding:** Determine the top 10% of heads that contribute the most to the SVA logit difference.
3. **Deliverable:**
* A table/list showing the **Top 10% Critical Heads** (Layer index, Head index, Causal Score).
* A **Python dictionary** representing the proposed "Protection Mask" (mapping layer indices to protected head indices).
* A brief summary of the complexity encountered and the percentage of total parameters covered by the mask.
