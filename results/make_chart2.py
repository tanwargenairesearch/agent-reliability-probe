"""Post 2 chart: three-bucket failure breakdown per dataset.

Stacked bars — works-every-time (green) / inconsistent (amber) / fails-every-time
(red) — showing that across models and domains, the inconsistent (intermittent)
slice dominates the failures.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (always_pass, intermittent, systematic) from the real n=100/50 runs
data = {
    "Retail · Model A":  (70, 23, 7),
    "Retail · Model B":  (72, 23, 5),
    "Airline · Model A": (29, 14, 7),
    "Airline · Model B": (35, 11, 4),
}
labels = list(data.keys())
# normalize to % so retail (100) and airline (50) compare on one axis
def pct(t):
    n = sum(t)
    return [100 * x / n for x in t]
vals = np.array([pct(data[k]) for k in labels])  # rows: dataset, cols: 3 buckets

green, amber, red = "#0d9488", "#e0a106", "#c2453b"
fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=200)
y = np.arange(len(labels))[::-1]  # top-to-bottom
left = np.zeros(len(labels))
segs = [
    ("Works every time (5/5)", green),
    ("Inconsistent (1–4 of 5)", amber),
    ("Fails every time (0/5)", red),
]
for i, (name, color) in enumerate(segs):
    ax.barh(y, vals[:, i], left=left, color=color, label=name, height=0.62)
    for j, v in enumerate(vals[:, i]):
        if v >= 5:
            ax.text(left[j] + v / 2, y[j], f"{v:.0f}%", ha="center", va="center",
                    color="white", fontsize=10.5, fontweight="bold")
    left += vals[:, i]

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlim(0, 100)
ax.set_xlabel("Share of tasks (%)", fontsize=10)
ax.set_title("How agents fail: most failures are inconsistent, not repeatable",
             fontsize=14, fontweight="bold", pad=30)
ax.text(0, 1.04, "150 tasks × 5 runs each · two frontier models · retail + airline · "
        "amber = the failures a single run can't see",
        transform=ax.transAxes, fontsize=9.5, color="#555")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=9.5)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(left=False)
ax.set_axisbelow(True)
fig.text(0.5, -0.02,
         "τ-bench retail + airline · database-state grading · pass^5 method · "
         "a complementary reliability view, not an official benchmark score · views my own",
         ha="center", fontsize=7.5, color="#999")
fig.tight_layout()
fig.savefig("results/post2_chart.png", bbox_inches="tight", facecolor="white")
print("wrote results/post2_chart.png")
