"""Render the Post 1 chart: accuracy (pass@1) vs reliability (pass^5).

Neutral, additive framing — both metrics shown as legitimate; the point is the
gap, not that either number is 'wrong'. Anonymized to match the post.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

models = ["Model A", "Model B"]
acc = [84.2, 84.8]   # pass@1
rel = [70.0, 72.0]   # pass^5

x = np.arange(len(models))
w = 0.34
fig, ax = plt.subplots(figsize=(9, 5.6), dpi=200)
b1 = ax.bar(x - w / 2, acc, w, label="Accuracy — succeeds at least once (pass@1)", color="#4C78A8")
b2 = ax.bar(x + w / 2, rel, w, label="Reliability — succeeds all 5/5 (pass^5)", color="#0d9488")

ax.set_ylim(0, 100)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel("Task success (%)", fontsize=11)
ax.set_title("Accuracy vs Reliability — 100 real customer-service agent tasks",
             fontsize=14, fontweight="bold", pad=34)
ax.text(0.5, 1.05, "Both models look capable on a single try — and less dependable every time",
        transform=ax.transAxes, ha="center", fontsize=10.5, color="#555")

for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.3,
                f"{b.get_height():.0f}%", ha="center", fontsize=11, fontweight="bold")

for i in range(2):
    gap = acc[i] - rel[i]
    ax.annotate("", xy=(x[i] + w / 2, rel[i]), xytext=(x[i] + w / 2, acc[i]),
                arrowprops=dict(arrowstyle="<->", color="#c2620b", lw=1.6))
    ax.text(x[i] + w / 2 + 0.07, (acc[i] + rel[i]) / 2, f"−{gap:.0f} pts",
            color="#c2620b", fontsize=10.5, fontweight="bold", va="center")

ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=1, frameon=False, fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
fig.text(0.5, -0.04,
         "τ-bench retail · database-state grading · 100 tasks × 5 runs · each model run as the agent\n"
         "A complementary reliability view, not an official benchmark score · views my own",
         ha="center", fontsize=7.5, color="#888")
fig.tight_layout()
fig.savefig("results/post1_chart.png", bbox_inches="tight", facecolor="white")
print("wrote results/post1_chart.png")
