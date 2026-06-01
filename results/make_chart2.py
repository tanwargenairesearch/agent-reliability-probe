"""Post 2 chart: zoom into the failures.

The story is "of the failures, ~3 in 4 are intermittent" — but failures are a
minority of outcomes, so a plain stacked bar buries them under green. This chart
shows all outcomes (left), then magnifies just the failing slice (right) so the
intermittent-dominates message actually pops.

Pooled across 2 models × 2 domains (retail+airline), 150 tasks/model, k=5.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

# pooled counts (300 task-outcomes = 150 tasks x 2 models)
WORKS, INTER, SYST = 206, 71, 23
TOTAL = WORKS + INTER + SYST          # 300
FAILS = INTER + SYST                  # 94
w_pct = 100 * WORKS / TOTAL
i_pct_all = 100 * INTER / TOTAL
s_pct_all = 100 * SYST / TOTAL
i_share = 100 * INTER / FAILS         # 75.5
s_share = 100 * SYST / FAILS          # 24.5

GREEN, AMBER, RED = "#0d9488", "#d98324", "#c2453b"
INK, SOFT = "#1f2328", "#5b6470"
plt.rcParams["font.family"] = "DejaVu Sans"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 6.2), dpi=200,
                               gridspec_kw={"width_ratios": [1, 1], "wspace": 0.45})
fig.patch.set_facecolor("white")

# ---- LEFT: all outcomes (one tall stacked bar) ----
xL = 0.0
bw = 0.5
axL.bar(xL, w_pct, bw, color=GREEN)
axL.bar(xL, i_pct_all, bw, bottom=w_pct, color=AMBER)
axL.bar(xL, s_pct_all, bw, bottom=w_pct + i_pct_all, color=RED)
axL.text(xL, w_pct / 2, f"Works every time\n{WORKS} of {TOTAL}\n({w_pct:.0f}%)",
         ha="center", va="center", color="white", fontsize=11, fontweight="bold")
axL.text(xL + bw/2 + 0.04, w_pct + (i_pct_all + s_pct_all) / 2,
         f"{FAILS} failed\non ≥1 run", ha="left", va="center", color=INK, fontsize=10.5)
axL.set_xlim(-0.55, 0.7)
axL.set_ylim(0, 100)
axL.set_title("Every outcome", fontsize=12.5, fontweight="bold", color=INK, pad=8)
axL.set_ylabel("Share of task-outcomes (%)", fontsize=10, color=SOFT)
axL.set_xticks([])
for s in ("top", "right", "bottom"):
    axL.spines[s].set_visible(False)
axL.tick_params(colors=SOFT)

# ---- RIGHT: zoom into just the failures (wide bar so labels fit) ----
xR = 0.0
bwR = 1.25
axR.bar(xR, i_share, bwR, color=AMBER)
axR.bar(xR, s_share, bwR, bottom=i_share, color=RED)
axR.text(xR, i_share / 2 + 6, f"{i_share:.0f}%", ha="center", va="center",
         color="white", fontsize=30, fontweight="bold")
axR.text(xR, i_share / 2 - 11, "INTERMITTENT", ha="center", va="center",
         color="white", fontsize=13, fontweight="bold")
axR.text(xR, i_share / 2 - 17, "right sometimes, wrong other times",
         ha="center", va="center", color="white", fontsize=9.5)
axR.text(xR, i_share + s_share / 2, f"Always fails · {s_share:.0f}%",
         ha="center", va="center", color="white", fontsize=11, fontweight="bold")
axR.set_xlim(-0.9, 0.9)
axR.set_ylim(0, 100)
axR.set_title("Just the failures, magnified", fontsize=12.5, fontweight="bold", color=INK, pad=8)
axR.set_xticks([])
axR.set_yticks([])
for s in ("top", "right", "bottom", "left"):
    axR.spines[s].set_visible(False)

# ---- zoom connector lines: top & bottom of the LEFT failing slice -> right bar edges ----
for yL, yR in [(w_pct, 0), (100, 100)]:
    con = ConnectionPatch(xyA=(xL + bw/2, yL), coordsA=axL.transData,
                          xyB=(xR - bwR/2, yR), coordsB=axR.transData,
                          color="#cbd2d9", lw=1.2, linestyle=(0, (4, 3)))
    fig.add_artist(con)

# ---- titles ----
fig.suptitle("3 of every 4 agent failures are the kind a single test can't see",
             fontsize=16.5, fontweight="bold", color=INK, x=0.5, y=1.0)
fig.text(0.5, 0.935,
         "Each task run 5 times. A failure that happens every time, you catch in QA. "
         "An intermittent one ships — then breaks for a real customer.",
         ha="center", fontsize=10.5, color=SOFT)
fig.text(0.5, -0.01,
         "1,500 runs · 2 frontier models × retail + airline (τ-bench) · database-state grading · "
         "pass^k method · complementary reliability view, not an official benchmark score · views my own",
         ha="center", fontsize=7.6, color="#9aa3ad")

fig.savefig("results/post2_chart.png", bbox_inches="tight", facecolor="white")
print("wrote results/post2_chart.png")
