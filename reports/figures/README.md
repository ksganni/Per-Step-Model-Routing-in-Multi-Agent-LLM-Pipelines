# Sprint 5 figures - how to read them

These plots are built from `experiments/qa_results.csv` by:

```bash
python reports/plot_figures.py
```

Re-run that command whenever the CSV gains new rows. PNG and PDF copies are written into this folder.

**Experiment idea in one sentence:** for the HotpotQA multi-hop pipeline, can we put a cheaper model on one step at a time without hurting the final answer much, and where do mistakes still reach the end?

The QA pipeline steps (in order) are:

1. **Decomposer** - splits the user question into smaller sub-questions  
2. **Hop reader** - reads evidence for each sub-question  
3. **Composer** - merges hop answers into one draft answer  
4. **Formatter** - produces the final short answer string  

Models compared:

- **Flash** - strong paid Gemini model (baseline / "all-strong")  
- **Flash-Lite** - cheaper Gemini model ("cheap-*" configs)  
- **Qwen** - local open model on Kaggle/ngrok ("qwen-*" configs)  

Each cheap/qwen config changes **only one** step; the other steps stay Flash.

---

## Core deliverable figures (Sprint 5 checklist)

### 1. `accuracy_cost_heatmap.png`

**What it is:** a grid with pipeline steps as rows and models as columns.

**What each cell means:**

- **Number on top** - mean F1 score for that setup  
- **(n=...)** - how many questions that cell is based on  
- **Color** - mean dollar cost per question (darker = more expensive)  

**How to read it:** compare Flash vs Flash-Lite vs Qwen across one row to see whether that step tolerates a weaker model.

---

### 2. `error_propagation_sankey.png`

**What it is:** a flow diagram about mistakes that survive to the final answer.

**Setup:** take questions that all-strong got right. For each cheap/qwen single-step run on the same ids:

- stayed correct -> **Final OK (no prop.)** (green)  
- became wrong -> **Final wrong (propagated)** (red)  

**Caveat:** this is a proxy based on end-to-end correctness (question-level CSV rows), not raw step-text diffs.

---

### 3. `routing_map.png`

**What it is:** the main recommendation picture - one model choice per step.

**Rule:** for each step, pick the cheapest model whose mean F1 is within 0.05 of all-strong F1.

---

### 4. `cost_savings_bars.png`

**What it is:** three bars - all-strong cost, recommended routing cost (estimate), all-cheap Lite proxy cost - with approximate F1 labels.

---

## Extra clarity figures (added for understanding)

### 5. `config_overview.png`

**What it is:** two panels for **every config name** in the CSV.

- Left: mean F1 (quality)  
- Right: mean cost  

**Why it helps:** you can see all nine experiments as simple bars before reading the heatmap. The dashed line is all-strong F1.

---

### 6. `f1_vs_cost_scatter.png`

**What it is:** one point per config - x = cost, y = F1. Marker size grows with n.

**Why it helps:** the "good" region is **up and left** (accurate and cheap). Points down-right are expensive or weak.

---

### 7. `f1_drop_by_step.png`

**What it is:** for each step, how much F1 **changes** when that step alone is swapped to Flash-Lite or Qwen (relative to all-strong).

**Why it helps:** bars near zero mean "this step is safe to cheapen." The red dashed line is the -0.05 F1 tolerance used by the routing map.

---

### 8. `sample_coverage.png`

**What it is:** how many questions finished for each config (out of the 150 target).

**Why it helps:** uneven n means some heatmap cells are less trustworthy. Use this before over-claiming results.

---

### 9. `propagation_rate_bars.png`

**What it is:** a simpler view of the Sankey idea - for each weak step/model, the **percent** of baseline-correct questions that become wrong.

**Why it helps:** easier to quote in text ("X% propagation on hop reader") than reading ribbon thickness.

---

### 10. `em_vs_f1_by_config.png`

**What it is:** Exact Match rate vs mean F1 for each config.

**Why it helps:** EM is strict (must match gold exactly). F1 gives partial credit. If F1 is high but EM is low, answers are close but not exact.

---

## Companion file: `figure_summary.txt`

Plain-text snapshot from every plot run: CSV path, row count, all-strong F1/cost, recommended routing, and the figure file list.

---

## Refresh checklist

1. Add more rows to `experiments/qa_results.csv` (resume `run_qa_matrix.py`).  
2. Run `python reports/plot_figures.py`.  
3. Open the PNGs (or PDFs for the report).  
4. Skim `figure_summary.txt` for the updated recommendation line.  
