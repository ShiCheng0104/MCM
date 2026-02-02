# Our Work Flowchart — Drawing Plan (≤4000 words)

This document is a **ready-to-draw** plan for an “Our Work” flowchart that is **lively, story-driven, and clearly highlights our contributions**. It is tailored to this project’s pipeline: data collection → vote inference → method comparison → factor analysis → new system design → evaluation & sensitivity.

---

## 1) One-sentence purpose (put above the figure)

**Figure X. Our end-to-end workflow: we reconstruct hidden audience votes from partial information, validate them against real eliminations, and use the inferred votes to compare rules, explain drivers, and design a better voting system.**

Alternative (more “four-task” explicit):
**Figure X. Our four-task workflow: infer the hidden audience votes (Task 1), run counterfactual rule comparisons (Task 2), identify drivers behind judges vs audience preferences (Task 3), and design + backtest an improved voting system (Task 4).**

---

## 2) Visual style guide

### 2.1 Color palette (use the project palette)

From [配色.md](../配色.md):

- **Deep Navy** `#264653` — titles, borders, anchors (“Ground Truth” / “Constraints”)
- **Teal** `#2A9D8E` — data pipeline blocks (“Collect”, “Clean”, “Merge”)
- **Sand** `#E9C46B` — modeling blocks (“Infer”, “Optimize”, “Monte Carlo”)
- **Orange** `#F3A261` — analysis blocks (“Compare”, “Explain”, “Predict”)
- **Coral** `#E86F52` — final outputs & recommendations (“System”, “Policy”, “Insights”)

Accessibility tips:

- Keep text in **Navy** on light fills; use white text only for Navy/Teal fills.
- Use consistent icon + label patterns to avoid visual clutter.

### 2.2 Typography & shapes

- Font: **Inter / Arial** (clean), 10–11pt inside boxes.
- Boxes: rounded rectangles; **major phases** slightly larger.
- Arrows: solid for “main pipeline”, dashed for “feedback loops / validation”.
- Add small icons (optional) for friendliness: database, broom, sigma, scales, magnifier, trophy.

### 2.3 Layout choice (recommended)

A **compact, four-task panel layout** (inspired by typical MCM “task blocks” figures):

- Use a **2×2 grid** for **Task 1–Task 4** (four colored panels).
- Place **one shared input box** above the grid and **one shared output box** below.
- Make arrows short and mostly vertical; use **one central “hub arrow”** from Task 1 into Tasks 2–4.

This keeps the figure tight (no long horizontal flow) while clearly showing **task boundaries, internal steps, and task-to-task interactions**.

---

## 3) Flowchart content (exact boxes to draw, ≤9 boxes)

Design goal: **≤9 boxes**, but visually structured as **four task panels** with internal steps.

### 3.1 Box list (recommended: 6 boxes total)

**Box A — Shared Inputs (Data Hub)** (Teal)

- judges’ scores + eliminations (weekly)
- contestant attributes (age / industry / partner)
- external signals: Google Trends + TV ratings
- cleaning & alignment: names, weeks, missing seasons

Then place the four task panels in a **2×2 grid**:

**Box T1 — Task 1: Hidden Audience Vote Inference** (Sand, largest panel)
Inside (show as 3 stacked mini-steps inside the same panel):

1) **Prior (belief)**: power-law score–vote link + popularity + preference
   - optional formula (tiny): $\hat{V}_i \propto S_i^{\alpha}(1+\theta_i)(1+w_p p_i)(1+w_r r_i)$
2) **Posterior (constraint inversion)**:
   - match real eliminations; minimal deviation from prior
   - SoftRank for rank-rule differentiability
   - SLSQP optimization (regularized)
3) **Uncertainty**: Monte Carlo → CI; CV as stability

**Box T2 — Task 2: Rule Comparison & Controversy** (Orange panel)
Inside (2–3 mini-steps):

- counterfactual simulator: rank vs percent rule outcomes
- disagreement diagnostics: “which is more audience-favoring?”
- controversy spotlight: 4 iconic contestants (trajectory + what-if)

**Box T3 — Task 3: Driver Analysis (Judges vs Audience)** (Orange panel)
Inside (2–3 mini-steps):

- effect estimation: OLS / mixed effects (marginal impacts)
- ML importance: random forest feature importance
- optional interpretability: SHAP-style explanations

**Box T4 — Task 4: New Voting System Design + Backtesting** (Coral panel)
Inside (2–3 mini-steps):

- dynamic weights by stage + technical protection (early weeks)
- controversy trigger + judge save (keep suspense but controlled)
- backtest scorecard: fairness / entertainment / controversy rate / simplicity

**Box B — Shared Outputs (What we deliver)** (Coral, bottom)

- inferred votes + uncertainty (reproducible tables/figures)
- evidence-based rule recommendation + tradeoffs
- proposed system + backtesting report

### 3.2 Interactions (must show with arrows)

Keep arrows short and explicit:

**Input arrows**

- A → T1, A → T3 (data and signals feed inference + driver analysis)

**Task coupling arrows (the key story)**

- T1 → T2 labeled “inferred votes”
- T1 → T3 labeled “inferred votes”
- T1 → T4 labeled “votes + uncertainty constraints”
- T2 → T4 labeled “what works / where it fails”
- T3 → T4 labeled “drivers → design knobs”

**Feedback loop (one dashed arrow, optional)**

- T4 → T1 (dashed) labeled “recalibrate weights / priors (sanity checks)”

Finally: T2, T3, T4 → B (three arrows converging).

---

## 4) “Make it vivid” details (small, not more boxes)

### 4.1 Micro-story labels (tiny captions near arrows)

- A → T1: “signals → beliefs”
- T1 (prior → posterior): “beliefs → constraint-consistent votes”
- T1 → T2: “reconstructed votes → rule experiments”
- T2/T3 → T4: “evidence + drivers → design knobs”

### 4.2 One “hero badge” (optional)

Put one small badge near T1 (top-right corner of the panel):

- “Core contribution: reconstructing hidden votes under real constraints.”

### 4.3 One feedback loop (shows scientific rigor)

Use **only one** dashed arrow (avoid clutter):

- T4 → T1: “iterate design & recalibrate”

---

## 5) How to produce the figure (two practical options)

### Option A (fast, recommended): diagrams.net (draw.io)

1. Create canvas: **A4 portrait** (recommended for a compact, non-wide figure).
2. Draw one top box (A), a centered 2×2 grid (T1–T4), and one bottom box (B).
3. Inside each task panel, place 2–3 mini-steps as smaller rounded rectangles **without counting them as separate “main boxes”** (they are inside the task panel).
4. Add arrows exactly as in Section 3.2; keep labels short.
5. Export:
   - PDF (vector) for LaTeX inclusion, and
   - PNG (300 dpi) for quick preview.

### Option B (LaTeX-native): TikZ

- Best if you want perfect font consistency with the paper.
- Export as PDF and include with `\includegraphics`.

---

## 6) Where to place it in the paper

- Put this figure at the start of your modeling section, right after the introduction/problem restatement.
- Refer to it when transitioning between Q1→Q2→Q3→Q4 (“as shown in Figure X”).

---

## 7) Suggested short caption (pick one)

**Caption A (formal)**: “End-to-end workflow of data enrichment, audience vote inference under elimination constraints, counterfactual rule comparison, factor analysis, and new voting system design with backtesting.”

**Caption B (more lively)**: “From signals to decisions: we fuse popularity and performance to infer hidden audience votes, test alternative rules in counterfactual simulations, explain what drives outcomes, and propose a more engaging yet fair system.”

**Caption C (four-task framing)**: “Four-task workflow with explicit interactions: Task 1 infers hidden audience votes under real elimination constraints; Tasks 2–3 analyze rule effects and preference drivers; Task 4 synthesizes evidence into a new system and backtests it with a scorecard.”
