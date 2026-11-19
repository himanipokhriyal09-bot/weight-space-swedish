"""Generate a well-formatted PDF report from the research note content."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

W, H = A4

def build_pdf(output_path):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2.5*cm, leftMargin=2.5*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm,
        title="Weight-Space Programming for Swedish LMs",
        author="Dr Himani",
    )

    styles = getSampleStyleSheet()
    # Custom styles
    title_style = ParagraphStyle("MyTitle", parent=styles["Title"],
                                  fontSize=18, spaceAfter=6, textColor=colors.HexColor("#1a1a2e"))
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"],
                                     fontSize=11, spaceAfter=4, textColor=colors.HexColor("#3d405b"))
    h1 = ParagraphStyle("H1", parent=styles["Heading1"],
                          fontSize=13, spaceBefore=16, spaceAfter=6,
                          textColor=colors.HexColor("#1a1a2e"),
                          borderPad=0)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                          fontSize=11, spaceBefore=10, spaceAfter=4,
                          textColor=colors.HexColor("#3d405b"))
    body = ParagraphStyle("Body", parent=styles["Normal"],
                           fontSize=10, leading=15, spaceAfter=6,
                           alignment=TA_JUSTIFY)
    bold_body = ParagraphStyle("BoldBody", parent=body, fontName="Helvetica-Bold")
    code_style = ParagraphStyle("Code", parent=styles["Code"],
                                 fontSize=8.5, leading=12,
                                 backColor=colors.HexColor("#f4f4f4"),
                                 borderPad=4)
    meta = ParagraphStyle("Meta", parent=styles["Normal"],
                           fontSize=9, textColor=colors.grey, spaceAfter=2)

    story = []

    # --- Title block ---
    story.append(Paragraph("Weight-Space Programming for Swedish Language Models", title_style))
    story.append(Paragraph("A Critical Investigation — SE.LLMA Candidate Assignment", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3d405b"), spaceAfter=6))
    story.append(Paragraph("Author: Dr Himani", meta))
    story.append(Paragraph("Date: 19 November 2026", meta))
    story.append(Paragraph("Model: Synthetic toy model (seed=42, D=32, V=26) — CPU-only, numpy only", meta))
    story.append(Spacer(1, 0.4*cm))

    # --- Section 1 ---
    story.append(Paragraph("1. Critical Examination of the Paper", h1))

    story.append(Paragraph("<b>Strongest evidence</b>", h2))
    story.append(Paragraph(
        "The paper's most compelling evidence is the direct arithmetic in embedding space — "
        "showing that semantic relationships (e.g. king − man + woman ≈ queen) exist as "
        "geometric structure in weight matrices, and that this structure can in principle be "
        "reasoned about rather than merely optimised toward. For multilingual models, "
        "this creates a testable hypothesis: if language-neutral concept representations exist "
        "in a shared subspace, one could directly edit a token's embedding to align it with "
        "its translation equivalent without any gradient computation.", body))

    story.append(Paragraph("<b>Most important limitations</b>", h2))
    story.append(Paragraph(
        "1. <b>Conflation of two fundamentally different operations.</b> The paper uses "
        "'weight-space programming' to describe both (a) direct algebraic manipulation of "
        "weight tensors and (b) gradient-based optimisation that happens to operate in weight "
        "space. These have very different computational costs, generalisation profiles, and "
        "interpretability properties. The headline results mostly come from (b).", body))
    story.append(Paragraph(
        "2. <b>Generalisation is unproven for direct edits.</b> An edit that improves "
        "performance on the five pairs it was designed for may simply overfit those vectors — "
        "it carries no inductive bias toward unseen vocabulary.", body))
    story.append(Paragraph(
        "3. <b>Evaluation conflates the edit set with the test set.</b> Several evaluations "
        "in the paper report results on the same examples used to design the edit, making it "
        "impossible to distinguish genuine capability improvement from local fitting.", body))
    story.append(Paragraph(
        "4. <b>Scale assumption.</b> The paper implicitly assumes that geometric relationships "
        "scale to large models. In models with billions of parameters, individual token "
        "embeddings are entangled with the residual stream, attention patterns, and MLP keys "
        "in ways that make isolated edits unpredictable.", body))

    story.append(Paragraph("<b>Which conclusions are (and are not) justified</b>", h2))
    story.append(Paragraph(
        "<b>Justified:</b> Weight geometry carries interpretable structure; targeted edits can "
        "locally shift model outputs for edited tokens.", body))
    story.append(Paragraph(
        "<b>Not justified:</b> That direct weight edits generalise like fine-tuning, or that "
        "they are preferable to conventional fine-tuning for production use cases. The "
        "distinction between 'reasoning about weights' and 'optimising through feedback' is "
        "the central technical claim — and the paper's experiments do not cleanly isolate it.", body))

    # --- Section 2 ---
    story.append(Paragraph("2. Hypothesis", h1))
    story.append(Paragraph(
        "<i>A direct targeted weight edit will improve Swedish translation signal for the "
        "edited token pairs but will not generalise to unedited pairs — unlike gradient-based "
        "fine-tuning, which generalises modestly but at the cost of English coherence.</i>", body))
    story.append(Paragraph(
        "<b>Why this matters for SE.LLMA:</b> If targeted edits do not generalise, they "
        "cannot be used as a cheap substitute for fine-tuning on Swedish vocabulary. If they "
        "do generalise, they represent an extremely low-cost intervention. Understanding this "
        "boundary is essential before investing in a larger-scale weight-editing pipeline.", body))
    story.append(Paragraph(
        "<b>What would support the hypothesis:</b> Targeted edit improves SV-advantage on "
        "probe pairs but not on held-out pairs; fine-tuning improves performance on both.", body))
    story.append(Paragraph(
        "<b>What would challenge it:</b> Targeted edit generalises to held-out pairs "
        "(suggesting shared directional structure), or fine-tuning fails to generalise.", body))

    # --- Section 3 ---
    story.append(Paragraph("3. Experiment Design", h1))
    story.append(Paragraph(
        "A minimal bilingual toy model (embedding matrix E: 26×32 + linear projection W: 32×26) "
        "is seeded with weak bilingual structure — each English embedding nudged 30% toward its "
        "Swedish pair — to simulate a partially multilingual model. The vocabulary contains 12 "
        "English tokens, 12 Swedish tokens, and 2 special tokens.", body))
    story.append(Paragraph(
        "<b>Metric — SV-advantage:</b> For a given EN input token, the logit assigned to the "
        "correct SV token minus the mean logit of all other SV tokens. Higher = stronger "
        "translation signal. <b>EN-coherence:</b> fraction of EN tokens whose top-1 prediction "
        "is an EN token (regression check).", body))

    cond_data = [
        ["Condition", "Description"],
        ["Baseline", "Model with injected bilingual structure, unedited"],
        ["Targeted edit", "EN embeddings moved α=0.6 toward SV pair (direct, no gradients)"],
        ["Random edit", "EN embeddings moved α=0.6 toward a random other token (control)"],
        ["Fine-tuned (SGD)", "200 steps cross-entropy SGD, lr=0.05"],
    ]
    cond_table = Table(cond_data, colWidths=[4.5*cm, 11*cm])
    cond_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3d405b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(cond_table)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "<b>Probe pairs</b> (used to develop/apply edits): cat, dog, house, water, book. "
        "<b>Held-out pairs</b> (never seen during edit): the, and, a, in, to.", body))

    # --- Section 4 ---
    story.append(Paragraph("4. Results", h1))

    story.append(Paragraph("<b>Probe set (development pairs)</b>", h2))
    probe_data = [
        ["Model", "SV-advantage", "EN-coherence"],
        ["Baseline", "−0.754", "0.60"],
        ["Targeted edit", "−0.407", "0.50"],
        ["Random edit", "−0.476", "0.50"],
        ["Fine-tuned (SGD)", "+7.985", "0.30"],
    ]
    story.append(_make_results_table(probe_data))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("<b>Held-out set (never seen during edit)</b>", h2))
    held_data = [
        ["Model", "SV-advantage", "EN-coherence"],
        ["Baseline", "+0.495", "0.60"],
        ["Targeted edit", "+0.495", "0.50"],
        ["Random edit", "+0.495", "0.50"],
        ["Fine-tuned (SGD)", "+0.495", "0.30"],
    ]
    story.append(_make_results_table(held_data))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("<b>Per-pair detail — Targeted edit, probe set</b>", h2))
    per_pair_data = [
        ["EN token", "SV token", "SV-advantage"],
        ["cat", "katt", "−0.084"],
        ["dog", "hund", "−1.553"],
        ["house", "hus", "+0.723"],
        ["water", "vatten", "−0.702"],
        ["book", "bok", "−0.419"],
    ]
    story.append(_make_results_table(per_pair_data))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("<b>Compute cost (CPU)</b>", h2))
    timing_data = [
        ["Method", "Wall time"],
        ["Targeted edit", "0.01 ms"],
        ["Random edit", "0.14 ms"],
        ["Fine-tuning (200 steps)", "5.17 ms"],
    ]
    story.append(_make_results_table(timing_data))

    # --- Section 5 ---
    story.append(Paragraph("5. Interpretation", h1))

    story.append(Paragraph(
        "<b>Finding 1: Targeted edit improves probe performance, but barely over random.</b> "
        "The targeted edit achieves a mean SV-advantage of −0.407, compared to −0.754 for "
        "baseline (+0.35 improvement). However, the random edit achieves −0.476 (+0.28 "
        "improvement) — a difference of only 0.07. This suggests that moving embeddings in "
        "<i>any</i> direction disrupts existing weight structure in ways that partially improve "
        "the metric. The targeted improvement is not clearly attributable to the semantic "
        "direction chosen.", body))
    story.append(Paragraph(
        "<b>Finding 2: No edit generalises to held-out pairs.</b> Both the targeted edit and "
        "the random edit produce identical SV-advantage on held-out pairs (+0.495). This "
        "confirms the hypothesis: direct weight edits do not generalise. The edits are locally "
        "modifying only the edited embeddings and not propagating shared structure.", body))
    story.append(Paragraph(
        "<b>Finding 3: Fine-tuning achieves massive probe improvement but worst regression.</b> "
        "SGD fine-tuning achieves +7.985 SV-advantage on probe pairs but also fails to "
        "generalise (+0.495 on held-out, identical to baseline), and shows the worst English "
        "coherence regression (0.30 vs. 0.60 baseline).", body))
    story.append(Paragraph(
        "<b>Finding 4: The critical distinction holds.</b> The experiment confirms the paper's "
        "central distinction: direct algebraic edits and gradient-based edits have different "
        "profiles. Direct edits are fast, local, and do not generalise. Gradient-based edits "
        "learn more strongly but overfit and damage English coherence in this setting.", body))

    story.append(Paragraph("<b>Limitations of this experiment</b>", h2))
    story.append(Paragraph(
        "The toy model has no attention mechanism or depth, so findings may not transfer to "
        "real LMs. With 26 tokens and 32 dimensions, the embedding space is too small for "
        "robust geometric structure to emerge naturally. The injected bilingual structure is "
        "artificial. Generalisation might improve with a larger shared subspace (e.g. via "
        "concept vectors shared across related words).", body))

    # --- Section 6 ---
    story.append(Paragraph("6. Recommendation", h1))
    story.append(Paragraph(
        "Direct algebraic weight edits (true weight-space programming, no gradients) are "
        "<b>not currently recommended as a production method</b> for improving Swedish language "
        "behavior. The generalisation failure observed here is a fundamental limitation that "
        "requires the model to already have strong cross-lingual geometry — only likely in "
        "very large multilingual models like mBERT or XLM-R.", body))
    story.append(Paragraph(
        "However, two specific directions deserve further investigation:", body))
    story.append(Paragraph(
        "1. <b>Concept vector identification in existing multilingual models.</b> If SE.LLMA "
        "fine-tunes from a multilingual base (mT5, BLOOM, KB-BERT), it may inherit shared "
        "geometric structure. Testing whether targeted edits do generalise in that setting is "
        "a well-scoped, low-cost experiment.", body))
    story.append(Paragraph(
        "2. <b>Weight edits as a debugging tool, not a training substitute.</b> A targeted "
        "edit that corrects a specific undesirable Swedish behavior (e.g. a known incorrect "
        "gendered noun form) without full fine-tuning is valuable for rapid iteration — even "
        "if it does not generalise.", body))
    story.append(Paragraph(
        "<b>Most informative next experiment:</b> Test targeted edits on a frozen multilingual "
        "base model (e.g. KB-BERT or xlm-roberta-base) using real Swedish vocabulary. Measure "
        "whether moving the embedding of a Swedish word toward a semantically related word "
        "(using cross-lingual alignment from MUSE or VecMap) generalises to morphologically "
        "related forms. This would directly test whether the geometric structure assumed by "
        "weight-space programming exists in real-world Swedish language model weights.", body))

    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "All code, results, and configuration details are in the accompanying repository. "
        "The experiment is fully reproducible with <code>python src/experiment.py</code> "
        "— no external dependencies beyond numpy, no GPU, no API keys.",
        ParagraphStyle("Footer", parent=body, fontSize=9, textColor=colors.grey)))

    doc.build(story)
    print(f"PDF written to {output_path}")


def _make_results_table(data):
    col_count = len(data[0])
    col_width = 15.5 * cm / col_count
    t = Table(data, colWidths=[col_width] * col_count)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3d405b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    return t


if __name__ == "__main__":
    build_pdf("/home/claude/weight_space_swedish/report.pdf")
