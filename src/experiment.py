"""
Weight-Space Programming for Swedish Language Models
Experiment: Can a targeted weight edit improve Swedish token prediction
without degrading English capability?

This experiment uses a minimal transformer-like weight structure
(embedding + single attention + FFN) to test the core claim of
weight-space programming in a controlled, reproducible setting.

Resources: CPU-only, numpy only. No API keys required.
Author: Dr Himani (candidate)
"""

import numpy as np
import json
import time
from copy import deepcopy

np.random.seed(42)

# ---------------------------------------------------------------------------
# Synthetic vocabulary & data
# ---------------------------------------------------------------------------

# A minimal bilingual vocabulary (English + Swedish function words + tokens)
VOCAB = {
    # English
    "the": 0, "is": 1, "and": 2, "a": 3, "in": 4, "to": 5, "of": 6,
    "cat": 7, "dog": 8, "house": 9, "water": 10, "book": 11,
    # Swedish
    "den": 12, "är": 13, "och": 14, "en": 15, "i": 16, "till": 17, "av": 18,
    "katt": 19, "hund": 20, "hus": 21, "vatten": 22, "bok": 23,
    # Shared/neutral
    "<PAD>": 24, "<UNK>": 25,
}
ID2TOK = {v: k for k, v in VOCAB.items()}
V = len(VOCAB)  # vocabulary size

# Translation pairs (EN -> SV): these define the "correct" behavior we want
EN_SV_PAIRS = [
    ("cat",   "katt"),
    ("dog",   "hund"),
    ("house", "hus"),
    ("water", "vatten"),
    ("book",  "bok"),
    ("the",   "den"),
    ("and",   "och"),
    ("a",     "en"),
    ("in",    "i"),
    ("to",    "till"),
]

# Probe contexts: given EN word in a "translate to Swedish" context,
# model should predict SV equivalent with higher probability.
# We test this by measuring logit of correct SV token vs. random SV token.

PROBE_PAIRS = EN_SV_PAIRS[:5]   # used to DEVELOP / SELECT the edit
HELD_OUT_PAIRS = EN_SV_PAIRS[5:]  # held out — never seen during edit


# ---------------------------------------------------------------------------
# Minimal model: embedding lookup + single linear layer (output logits)
# ---------------------------------------------------------------------------

class TinyBilingualModel:
    """
    Toy model: embedding matrix E (V x D) + output projection W (D x V).
    Forward pass: embed token, project to vocabulary logits.
    Stands in for the final layer behavior of a real LM.
    """

    def __init__(self, vocab_size=V, dim=32, seed=0):
        rng = np.random.default_rng(seed)
        # Random initialisation — no pretraining, so we inject structure manually
        self.E = rng.normal(0, 0.5, (vocab_size, dim)).astype(np.float32)
        self.W = rng.normal(0, 0.5, (dim, vocab_size)).astype(np.float32)
        self.dim = dim
        self.vocab_size = vocab_size

    def forward(self, token_id: int) -> np.ndarray:
        """Return logits over vocabulary for a given input token."""
        emb = self.E[token_id]           # (D,)
        logits = emb @ self.W            # (V,)
        return logits

    def predict_top(self, token_id: int, k: int = 3):
        logits = self.forward(token_id)
        topk = np.argsort(logits)[::-1][:k]
        return [(ID2TOK[i], float(logits[i])) for i in topk]

    def copy(self):
        m = TinyBilingualModel.__new__(TinyBilingualModel)
        m.E = self.E.copy()
        m.W = self.W.copy()
        m.dim = self.dim
        m.vocab_size = self.vocab_size
        return m


# ---------------------------------------------------------------------------
# Inject weak bilingual structure into the baseline model
# (simulates a partially-trained model that has SOME cross-lingual signal)
# ---------------------------------------------------------------------------

def inject_bilingual_structure(model: TinyBilingualModel, strength: float = 0.3):
    """
    For each EN-SV pair, nudge the EN embedding slightly toward the SV embedding.
    This creates a weak but learnable alignment — analogous to a model that has
    seen mixed-language data but not fine-tuned for translation.
    """
    m = model.copy()
    for en_w, sv_w in EN_SV_PAIRS:
        en_id = VOCAB[en_w]
        sv_id = VOCAB[sv_w]
        direction = m.E[sv_id] - m.E[en_id]
        m.E[en_id] += strength * direction
    return m


# ---------------------------------------------------------------------------
# Metric: translation logit advantage
# P(correct SV | EN input) - mean P(other SV tokens | EN input)
# Higher = model more strongly predicts the correct Swedish word
# ---------------------------------------------------------------------------

SV_TOKEN_IDS = [VOCAB[w] for _, w in EN_SV_PAIRS]

def translation_logit_advantage(model: TinyBilingualModel, pairs) -> dict:
    results = {}
    for en_w, sv_w in pairs:
        en_id = VOCAB[en_w]
        sv_id = VOCAB[sv_w]
        logits = model.forward(en_id)
        correct_logit = logits[sv_id]
        other_sv_logits = [logits[VOCAB[w]] for _, w in EN_SV_PAIRS if w != sv_w]
        advantage = float(correct_logit - np.mean(other_sv_logits))
        results[en_w] = {
            "correct_sv": sv_w,
            "correct_logit": float(correct_logit),
            "mean_other_sv_logit": float(np.mean(other_sv_logits)),
            "advantage": advantage,
        }
    return results


def english_coherence_score(model: TinyBilingualModel) -> float:
    """
    Regression check: for each EN word, top-1 prediction should NOT be a SV word.
    Score = fraction of EN words whose top-1 prediction is an EN word.
    """
    EN_IDS = set(range(12))   # English token ids 0-11
    correct = 0
    for en_w in [p[0] for p in EN_SV_PAIRS]:
        en_id = VOCAB[en_w]
        logits = model.forward(en_id)
        top1 = int(np.argmax(logits))
        if top1 in EN_IDS:
            correct += 1
    return correct / len(EN_SV_PAIRS)


# ---------------------------------------------------------------------------
# Targeted weight edit (the "weight-space programming" approach)
# We directly modify embedding vectors to strengthen SV alignment.
# This is *direct weight reasoning*, NOT gradient-based optimization.
# ---------------------------------------------------------------------------

def targeted_weight_edit(model: TinyBilingualModel, pairs, alpha: float = 0.6):
    """
    For each (EN, SV) pair, move the EN embedding toward the SV embedding
    by factor alpha.  This is a rank-1 update, applied directly in weight space.

    Key distinction from fine-tuning:
      - No model rollouts / loss computation
      - No gradient descent
      - Single pass, O(D) per pair
    """
    m = model.copy()
    for en_w, sv_w in pairs:
        en_id = VOCAB[en_w]
        sv_id = VOCAB[sv_w]
        sv_emb = m.E[sv_id]
        en_emb = m.E[en_id]
        delta = sv_emb - en_emb
        m.E[en_id] = en_emb + alpha * delta
    return m


# ---------------------------------------------------------------------------
# Alternative 1: Random / untargeted edit (control)
# ---------------------------------------------------------------------------

def random_weight_edit(model: TinyBilingualModel, pairs, alpha: float = 0.6, seed: int = 99):
    """Edit EN embeddings toward RANDOM other embeddings (not their SV pair)."""
    rng = np.random.default_rng(seed)
    m = model.copy()
    all_ids = list(range(V))
    for en_w, sv_w in pairs:
        en_id = VOCAB[en_w]
        sv_id = VOCAB[sv_w]
        # pick a random token that is NOT the correct SV word
        candidates = [i for i in all_ids if i != en_id and i != sv_id]
        rand_id = rng.choice(candidates)
        delta = m.E[rand_id] - m.E[en_id]
        m.E[en_id] = m.E[en_id] + alpha * delta
    return m


# ---------------------------------------------------------------------------
# Alternative 2: Gradient-based fine-tuning (conventional approach)
# ---------------------------------------------------------------------------

def gradient_finetune(model: TinyBilingualModel, pairs,
                      lr: float = 0.05, steps: int = 100):
    """
    Standard SGD on cross-entropy loss: for each EN input, maximise P(SV output).
    This IS optimization through repeated model feedback — the contrast case.
    """
    m = model.copy()
    for step in range(steps):
        for en_w, sv_w in pairs:
            en_id = VOCAB[en_w]
            sv_id = VOCAB[sv_w]
            # Forward
            emb = m.E[en_id]
            logits = emb @ m.W
            # Softmax
            logits_shifted = logits - logits.max()
            exp_l = np.exp(logits_shifted)
            probs = exp_l / exp_l.sum()
            # Cross-entropy grad w.r.t. logits
            dL_dlogits = probs.copy()
            dL_dlogits[sv_id] -= 1.0
            # Gradient w.r.t. embedding
            dL_demb = m.W @ dL_dlogits
            m.E[en_id] -= lr * dL_demb
    return m


# ---------------------------------------------------------------------------
# Run everything and collect results
# ---------------------------------------------------------------------------

def run_experiment():
    print("=" * 65)
    print("Weight-Space Programming for Swedish LMs — Experiment")
    print("=" * 65)

    # 1. Build baseline model with weak bilingual structure
    base = TinyBilingualModel(seed=42)
    base = inject_bilingual_structure(base, strength=0.3)

    # 2. Apply edits
    t0 = time.perf_counter()
    targeted = targeted_weight_edit(base, PROBE_PAIRS, alpha=0.6)
    t_targeted = time.perf_counter() - t0

    t0 = time.perf_counter()
    random_edit = random_weight_edit(base, PROBE_PAIRS, alpha=0.6)
    t_random = time.perf_counter() - t0

    t0 = time.perf_counter()
    finetuned = gradient_finetune(base, PROBE_PAIRS, lr=0.05, steps=200)
    t_finetune = time.perf_counter() - t0

    # 3. Evaluate on PROBE pairs (development set)
    print("\n--- Probe set (used to develop edit) ---")
    models = {
        "Baseline":        base,
        "Targeted edit":   targeted,
        "Random edit":     random_edit,
        "Fine-tuned (SGD)": finetuned,
    }

    probe_results = {}
    for name, model in models.items():
        adv = translation_logit_advantage(model, PROBE_PAIRS)
        mean_adv = np.mean([v["advantage"] for v in adv.values()])
        coh = english_coherence_score(model)
        probe_results[name] = {"mean_advantage": float(mean_adv),
                                "en_coherence": float(coh),
                                "per_pair": adv}
        print(f"  {name:22s}  sv_advantage={mean_adv:+.3f}  en_coherence={coh:.2f}")

    # 4. Evaluate on HELD-OUT pairs
    print("\n--- Held-out set (never seen during edit) ---")
    held_results = {}
    for name, model in models.items():
        adv = translation_logit_advantage(model, HELD_OUT_PAIRS)
        mean_adv = np.mean([v["advantage"] for v in adv.values()])
        coh = english_coherence_score(model)
        held_results[name] = {"mean_advantage": float(mean_adv),
                               "en_coherence": float(coh),
                               "per_pair": adv}
        print(f"  {name:22s}  sv_advantage={mean_adv:+.3f}  en_coherence={coh:.2f}")

    # 5. Timing
    print("\n--- Compute cost ---")
    print(f"  Targeted edit:    {t_targeted*1000:.2f} ms")
    print(f"  Random edit:      {t_random*1000:.2f} ms")
    print(f"  Fine-tune (200s): {t_finetune*1000:.2f} ms")

    # 6. Per-pair detail for targeted edit
    print("\n--- Per-pair detail: Targeted edit (probe set) ---")
    for en_w, detail in probe_results["Targeted edit"]["per_pair"].items():
        print(f"  {en_w:8s} -> {detail['correct_sv']:8s}  "
              f"advantage={detail['advantage']:+.3f}")

    print("\n--- Per-pair detail: Targeted edit (held-out set) ---")
    for en_w, detail in held_results["Targeted edit"]["per_pair"].items():
        print(f"  {en_w:8s} -> {detail['correct_sv']:8s}  "
              f"advantage={detail['advantage']:+.3f}")

    # 7. Save results
    results = {
        "probe": probe_results,
        "held_out": held_results,
        "timing_ms": {
            "targeted": t_targeted * 1000,
            "random": t_random * 1000,
            "finetune_200steps": t_finetune * 1000,
        },
        "config": {
            "vocab_size": V,
            "dim": 32,
            "alpha_edit": 0.6,
            "ft_lr": 0.05,
            "ft_steps": 200,
            "seed": 42,
            "probe_pairs": PROBE_PAIRS,
            "held_out_pairs": HELD_OUT_PAIRS,
        }
    }

    # Serialise numpy floats
    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(x) for x in obj]
        return obj

    with open("/home/claude/weight_space_swedish/results/results.json", "w") as f:
        json.dump(to_serializable(results), f, indent=2)

    print("\n✓ Results saved to results/results.json")
    return results


if __name__ == "__main__":
    run_experiment()
