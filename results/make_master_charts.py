"""Master charts from the compiled data (single source of truth).

  chart A: accuracy (pass@1) vs reliability (pass^5) across all domain×model cells
  chart B: failure taxonomy (corrected) — pooled retail+airline
Reads results/master_dataset.json + results/taxonomy.json.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, TEAL, AMBER, RED, PURPLE = "#2f6f9f", "#0d9488", "#d98324", "#c2453b", "#7a5195"
INK, SOFT = "#1f2328", "#5b6470"
plt.rcParams["font.family"] = "DejaVu Sans"

# ---------- Chart A: reliability across cells ----------
m = json.load(open("results/master_dataset.json"))
cells = m["cells"]
labels = [f"{c['domain']}\n{c['model'].split('-')[0]}" for c in cells]
acc = [c["pass@1"] * 100 for c in cells]
rel = [c["pass^5"] * 100 for c in cells]
clean = [c["clean"] is True for c in cells]

x = np.arange(len(cells))
w = 0.38
fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
b1 = ax.bar(x - w/2, acc, w, label="Accuracy (pass@1) — right once", color=BLUE)
b2 = ax.bar(x + w/2, rel, w, label="Reliability (pass^5) — right every time", color=TEAL)
for i in range(len(cells)):
    ax.text(x[i]-w/2, acc[i]+1.5, f"{acc[i]:.0f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(x[i]+w/2, rel[i]+1.5, f"{rel[i]:.0f}", ha="center", fontsize=9.5, fontweight="bold")
    if not clean[i]:
        ax.text(x[i], 4, "lower\nbound*", ha="center", fontsize=7.5, color=RED, style="italic")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 105); ax.set_ylabel("Task success (%)", fontsize=10)
ax.set_title("The accuracy–reliability gap holds across domains and models",
             fontsize=15, fontweight="bold", pad=28)
ax.text(0.5, 1.04, "Every cell: succeeding once (blue) overstates succeeding every time (teal)",
        transform=ax.transAxes, ha="center", fontsize=10.5, color=SOFT)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
fig.text(0.5, -0.05,
         "* telecom failures still hit the 60-step cap, so telecom pass rates are lower bounds.  "
         "τ-bench · DB-state grading · k=5 · views my own",
         ha="center", fontsize=7.6, color="#9aa3ad")
fig.tight_layout()
fig.savefig("results/master_reliability.png", bbox_inches="tight", facecolor="white")
print("wrote results/master_reliability.png")

# ---------- Chart B: failure taxonomy ----------
tax = json.load(open("results/taxonomy.json"))["pooled"]
order = sorted(tax.items(), key=lambda kv: -kv[1])
names = [k for k, _ in order]
vals = [v for _, v in order]
tot = sum(vals)
colors = {"Acted with wrong arguments/values": RED,
          "Escalated to human without completing": PURPLE,
          "Stalled — never executed the action": AMBER,
          "Lost the thread / looped (hit step cap)": BLUE}
fig2, ax2 = plt.subplots(figsize=(10.5, 5.2), dpi=200)
y = np.arange(len(names))[::-1]
ax2.barh(y, [100*v/tot for v in vals], color=[colors.get(n, SOFT) for n in names], height=0.62)
for i, (n, v) in enumerate(order):
    pc = 100*v/tot
    ax2.text(pc+1, y[i], f"{pc:.0f}%  ({v})", va="center", fontsize=11, fontweight="bold", color=INK)
ax2.set_yticks(y); ax2.set_yticklabels(names, fontsize=11.5)
ax2.set_xlim(0, max(100*v/tot for v in vals)+12)
ax2.set_title("How agents fail: four modes, none dominant",
              fontsize=15, fontweight="bold", pad=26)
ax2.text(0, 1.05, f"{tot} failing trajectories · retail + airline · Gemini-3.5-flash · DB-state grading",
         transform=ax2.transAxes, fontsize=10, color=SOFT)
ax2.spines[["top", "right", "bottom"]].set_visible(False); ax2.set_xticks([]); ax2.tick_params(left=False)
fig2.text(0.5, -0.04,
          "Heuristic classifier (write-tool calls · escalation · step cap), validated on a manual pilot · "
          "directional, not human-verified per run · views my own",
          ha="center", fontsize=7.6, color="#9aa3ad")
fig2.tight_layout()
fig2.savefig("results/master_taxonomy.png", bbox_inches="tight", facecolor="white")
print("wrote results/master_taxonomy.png")
