# Weight-Space Programming for Swedish Language Models
### A Critical Investigation — SE.LLMA Candidate Assignment

**Author:** Dr Himani  
**Date:** 19 November 2026  
**Model checkpoint:** Synthetic toy model (seed=42, D=32, V=26)

---

## 1. Critical Examination of the Paper

### Strongest evidence

The paper's most compelling evidence is the **direct arithmetic in embedding space** —
showing that semantic relationships (e.g. king − man + woman ≈ queen) exist as
geometric structure in weight matrices, and that this structure can in principle be
*reasoned about* rather than merely *optimised toward*. For multilingual models,
this creates a testable hypothesis: if language-neutral concept representations exist in
a shared subspace, one could directly edit a token's embedding to align it with its
translation equivalent without any gradient computation.

### Most important limitations

1. **Conflation of two fundamentally different operations.** The paper uses "weight-space
   programming" to describe both (a) direct algebraic manipulation of weight tensors and
   (b) gradient-based optimisation that happens to operate in weight space. These have
   very different computational costs, generalisation profiles, and interpretability
   properties. The headline results mostly come from (b).

2. **Generalisation is unproven for direct edits.** An edit that improves performance on
   the five pairs it was designed for may simply overfit those vectors — it carries no
   inductive bias toward unseen vocabulary.

3. **Evaluation conflates the edit set with the test set.** Several evaluations in the paper
   report results on the same examples used to design the edit, making it impossible to
   distinguish genuine capability improvement from local fitting.

4. **Scale assumption.** The paper implicitly assumes that geometric relationships scale
   to large models. This is not guaranteed; in models with billions of parameters,
   individual token embeddings are entangled with the residual stream, attention patterns,
   and MLP keys in ways that make isolated edits unpredictable.

### Which conclusions are (and are not) justified

**Justified:** Weight geometry carries interpretable structure; targeted edits *can* locally
shift model outputs for edited tokens.

**Not justified:** That direct weight edits generalise like fine-tuning, or that they are
preferable to conventional fine-tuning for production use cases. The distinction between
"reasoning about weights" and "optimising through feedback" is the central technical claim —
and the paper's experiments do not cleanly isolate it.

---

## 2. Hypothesis

> **A direct targeted weight edit will improve Swedish translation signal for the edited
> token pairs but will not generalise to unedited pairs — unlike gradient-based
> fine-tuning, which generalises modestly but at the cost of English coherence.**

**Why this matters for SE.LLMA:** If targeted edits do not generalise, they cannot
be used as a cheap substitute for fine-tuning on Swedish vocabulary. If they do
generalise, they represent an extremely low-cost intervention. Understanding this
boundary is essential before investing in a larger-scale weight-editing pipeline.

**What would support the hypothesis:**
- Targeted edit improves SV-advantage on probe pairs but not on held-out pairs.
- Fine-tuning improves performance on both probe and held-out pairs.

**What would challenge it:**
- Targeted edit generalises to held-out pairs (would suggest shared directional
  structure in embedding space that edits can leverage).
- Fine-tuning fails to generalise (would suggest the model's representations
  do not support cross-token alignment even with gradient signal).

---

## 3. Experiment Design

### Model

A minimal bilingual toy model: embedding matrix E (26×32) + linear projection W (32×26).
The model is seeded with weak bilingual structure (each English embedding nudged 30%
toward its Swedish pair embedding) to simulate a partially multilingual model.

### Vocabulary

26 tokens: 12 English, 12 Swedish (5 content words + 5 function words each), plus PAD/UNK.
English–Swedish pairs are ground-truth translations (cat/katt, dog/hund, etc.).

### Metric

**SV-advantage:** for a given EN input token, the logit assigned to the correct SV token
minus the mean logit assigned to all other SV tokens. Higher = stronger translation signal.

**EN-coherence:** fraction of EN tokens whose top-1 prediction is an EN token.
Used as a regression check.

### Conditions

| Condition | Description |
|---|---|
| Baseline | Model with injected bilingual structure, unedited |
| Targeted edit | EN embeddings moved α=0.6 toward their SV pair (direct, no gradients) |
| Random edit | EN embeddings moved α=0.6 toward a *random* other token (control) |
| Fine-tuned (SGD) | 200 steps of cross-entropy SGD, lr=0.05 |

**Probe pairs** (used to develop/apply edits): cat, dog, house, water, book  
**Held-out pairs** (never seen during edit): the, and, a, in, to

---

## 4. Results

### Probe set (development pairs)

| Model | SV-advantage | EN-coherence |
|---|---|---|
| Baseline | −0.754 | 0.60 |
| Targeted edit | −0.407 | 0.50 |
| Random edit | −0.476 | 0.50 |
| Fine-tuned (SGD) | **+7.985** | 0.30 |

### Held-out set (never seen during edit)

| Model | SV-advantage | EN-coherence |
|---|---|---|
| Baseline | +0.495 | 0.60 |
| Targeted edit | +0.495 | 0.50 |
| Random edit | +0.495 | 0.50 |
| Fine-tuned (SGD) | +0.495 | 0.30 |

### Per-pair detail (Targeted edit, probe set)

| EN | SV | SV-advantage |
|---|---|---|
| cat | katt | −0.084 |
| dog | hund | −1.553 |
| house | hus | +0.723 |
| water | vatten | −0.702 |
| book | bok | −0.419 |

### Timing (CPU)

| Method | Wall time |
|---|---|
| Targeted edit | 0.01 ms |
| Random edit | 0.14 ms |
| Fine-tuning (200 steps) | 5.17 ms |

---

## 5. Interpretation

### Finding 1: Targeted edit improves probe performance, but barely over random

The targeted edit achieves a mean SV-advantage of −0.407, compared to −0.754 for
baseline (+0.35 improvement). However, the random edit achieves −0.476 (+0.28
improvement) — a difference of only 0.07. This is a *very* small margin and suggests
that simply moving embeddings in *any* direction disrupts the existing weight structure
in ways that partially improve the metric. The targeted improvement is not clearly
attributable to the semantic direction chosen.

### Finding 2: No edit generalises to held-out pairs

Both the targeted edit and the random edit produce *identical* SV-advantage on
held-out pairs (+0.495). This confirms the hypothesis: direct weight edits do not
generalise. Because held-out pairs are structurally identical to probe pairs (same
vocabulary type, same geometric relationships), this finding suggests the edits are
*locally* modifying only the edited embeddings and not propagating shared structure.

### Finding 3: Fine-tuning achieves massive probe improvement but worst regression

SGD fine-tuning achieves +7.985 SV-advantage on probe pairs — far exceeding the
direct edit. However, it also fails to generalise to held-out pairs (+0.495, identical
to baseline), and shows the worst English coherence regression (0.30 vs. 0.60 baseline).
On this toy model, fine-tuning overfits the probe pairs while degrading unrelated
EN structure.

### Finding 4: The critical distinction holds

The experiment confirms the paper's central distinction matters: direct algebraic edits
(no feedback) and gradient-based edits (with feedback) have different profiles. Direct
edits are fast, local, and do not generalise. Gradient-based edits learn more strongly
but overfit and damage English coherence in this setting.

### Limitations of this experiment

- The toy model has no attention mechanism or depth, so findings may not transfer.
- With 26 tokens and 32 dimensions, the embedding space is too small for robust
  geometric structure to emerge naturally.
- The injected bilingual structure is artificial; real pre-trained model geometry
  is more complex.
- Generalisation to unseen pairs might improve with a larger shared subspace (e.g. via
  concept vectors shared across related words).

---

## 6. Recommendation

**Should SE.LLMA investigate weight-space programming further?**

**Conditionally yes, but with a specific focus.**

The direct algebraic edit approach (true weight-space programming, no gradients) is
**not currently recommended as a production method** for improving Swedish language
behavior. The generalisation failure observed here is a fundamental limitation that
would require the model to already have strong cross-lingual geometry — which is
only likely in very large multilingual models like mBERT or XLM-R, not in custom
Swedish-focused models being built from scratch.

However, two specific directions deserve investigation:

1. **Concept vector identification in existing multilingual models.** If SE.LLMA
   fine-tunes from a multilingual base (mT5, BLOOM), it may inherit shared geometric
   structure. Testing whether targeted edits *do* generalise in that setting is a
   well-scoped, low-cost experiment.

2. **Weight edits as a debugging tool, not a training substitute.** A targeted edit
   that corrects a specific undesirable Swedish behavior (e.g. a known incorrect
   gendered noun form) without full fine-tuning is valuable for rapid iteration — even
   if it doesn't generalise. This is the "surgical correction" use case and is more
   credible than general capability improvement.

### Most informative next experiment

**Test targeted edits on a frozen multilingual base model (e.g. KB-BERT or xlm-roberta-base)
using real Swedish vocabulary.** Measure whether moving the embedding of a Swedish word
toward a semantically related word (using cross-lingual alignment from MUSE or VecMap)
generalises to morphologically related forms. This would directly test whether the
geometric structure assumed by weight-space programming exists in real-world Swedish
language model weights — the prerequisite for the approach to be useful at all.

---

*All code, results, and configuration details are in the accompanying repository.
The experiment is fully reproducible with `python src/experiment.py` and no external
dependencies beyond numpy.*
