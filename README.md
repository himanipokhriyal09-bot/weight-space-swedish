# Weight-Space Programming for Swedish Language Models

**Candidate assignment — SE.LLMA**

## Quick start

```bash
# No install required beyond Python 3.8+ and numpy
pip install numpy          # only dependency

python src/experiment.py   # runs the full experiment (~1 second, CPU-only)
# Results written to results/results.json
```

## Repository layout

```
├── src/
│   └── experiment.py        # full experiment (model, edits, evaluation)
├── results/
│   └── results.json         # all numeric outcomes, reproducible with seed=42
├── research_note.md         # concise Markdown report (~1-2 pages)
├── report.pdf               # PDF version of the research note
├── ai_usage.md              # description of AI tool use
└── README.md                # this file
```

## Experiment summary

**Hypothesis:** A direct targeted weight edit (no gradient descent) can improve a
model's Swedish translation signal for edited token pairs, but — unlike fine-tuning —
will *not* generalise to unedited pairs.

**Key finding:** The targeted edit achieves a +0.35 improvement in mean SV-advantage
on probe pairs vs. baseline, comparable to the random edit (+0.28), and both are
far outperformed by fine-tuning (+8.74) *on the probe set*. On held-out pairs,
none of the edits generalise. Fine-tuning shows the worst English coherence regression.

See `research_note.md` for full interpretation and recommendation.

## Reproducibility

All random seeds are fixed (`numpy.random.seed(42)`, `default_rng(42)` for the model,
`default_rng(99)` for the random edit). The experiment is fully deterministic.
No GPU, no API key, no internet access required.
