# AI Tool Usage

## Tools used

**Claude (Anthropic) — claude-sonnet-4-6**

Used as a coding and writing assistant throughout the assignment.

## Specific contributions

- **Experiment scaffolding:** Claude generated the initial structure of `experiment.py`
  based on a description of the hypothesis and desired metrics. All scientific decisions
  (what to measure, what controls to use, what constitutes a fair comparison) were made
  by me.

- **Code review:** Claude reviewed the experiment code for correctness before running,
  catching a missing directory creation step.

- **Writing assistance:** The research note was drafted by me and reviewed by Claude for
  clarity and structure. Claude did not contribute scientific claims or interpretations —
  those are entirely my own.

- **PDF generation:** Claude assisted with the Python/reportlab code for PDF generation.

## What AI did NOT do

- Claude did not make scientific judgments about the paper's claims.
- Claude did not select the hypothesis, choose the metrics, or interpret the results.
- All experimental design decisions (probe vs. held-out split, choice of alpha,
  number of fine-tuning steps, regression metric) were made by me.

## Reflection

Using Claude as a coding assistant accelerated implementation but did not substitute
for scientific reasoning. The key judgments — that generalisation is the right thing
to test, that random edits are the correct control, that English coherence is the
right regression check — required reading the paper critically and understanding
what a fair experiment looks like.
