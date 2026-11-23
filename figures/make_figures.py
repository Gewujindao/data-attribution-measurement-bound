"""Draw figures/summary.png: errors of the estimator E3 in the 9 new settings at two replay budgets.

All numbers are constants copied from files in this repository; the source of
each one is given in the comment next to it. Run from any directory:
    python figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#1f4e79"
GREY = "#9aa5b1"
INK = "#222222"
MUTED = "#5f6b76"
AXIS = "#444444"
GRID = "#e6e9ec"
FS_TITLE, FS_SUB, FS_TEXT = 13, 10, 8  # the only three font sizes used

# Pass criterion, see README:
# median LDS error <= 0.05 and median rank90 error <= 0.25 over 20 draws.
LDS_PASS = 0.05
R90_PASS = 0.25

# setting: (LDS err s=k/10, LDS err s=k/5, rank90 err s=k/10, rank90 err s=k/5), medians over 20 draws
SETTINGS = [
    ("A", 0.055, 0.038, 0.19, 0.07),    # results/estimator_test/output_summary.txt
    ("B", 0.038, 0.028, 0.12, 0.05),    # results/estimator_test/output_summary.txt
    ("C", 0.061, 0.038, 0.04, 0.02),    # results/estimator_test/output_summary.txt
    ("D", 0.042, 0.039, 0.09, 0.09),    # results/estimator_test/output_summary.txt
    ("E", 0.031, 0.025, 0.05, 0.03),    # results/estimator_test/output_summary.txt
    ("F", 0.044, 0.030, 0.15, 0.09),    # results/estimator_test/output_summary.txt
    ("H1", 0.070, 0.047, 0.27, 0.16),   # results/estimator_test/output_summary.txt
    ("H2", 0.048, 0.038, 0.08, 0.04),   # results/estimator_test/output_summary.txt
    ("H3", 0.022, 0.016, 0.06, 0.04),   # results/estimator_test/output_summary.txt
]
LDS_UNDECIDABLE = ("H3",)  # L*_S = 0.068 < 0.3, judged on rank90 only: results/estimator_test/output_summary.txt
PASS_K10 = 6   # results/estimator_test/output_summary.txt
PASS_K5 = 9    # results/estimator_test/output_summary.txt


def setup():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": FS_TEXT,
        "axes.linewidth": 0.6,
        "axes.edgecolor": AXIS,
        "xtick.direction": "out",
        "xtick.major.size": 3,
        "xtick.major.width": 0.6,
        "xtick.color": AXIS,
        "xtick.labelsize": FS_TEXT,
        "ytick.labelsize": FS_TEXT,
    })


def panel(fig, rect, letter, title, idx10, idx5, threshold, xmax, ticks, tickfmt, fmt, undecidable=()):
    """Bars in an axes at `rect` (figure coords); values in two right-aligned columns to its right."""
    ax = fig.add_axes(rect)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.6)

    n = len(SETTINGS)
    bar_h = 0.36
    ax.set_ylim(-n + 0.5, 0.5)
    ax.set_xlim(0, xmax)
    for i, row in enumerate(SETTINGS):
        for off, k, colour in ((bar_h / 2, idx10, GREY), (-bar_h / 2, idx5, BLUE)):
            if row[0] in undecidable:  # not judged on this term: outline only
                ax.barh(-i + off, row[k], height=bar_h * 0.85, facecolor="white", edgecolor=colour,
                        linewidth=0.8)
            else:
                ax.barh(-i + off, row[k], height=bar_h * 0.85, color=colour, linewidth=0)
    ax.axvline(threshold, color=INK, linestyle=(0, (3, 2)), linewidth=0.8, zorder=3)
    ax.text(threshold, 0.62, f"pass \u2264 {fmt.format(threshold)}", ha="center", va="bottom", color=INK,
            clip_on=False)
    ax.set_yticks([-i for i in range(n)])
    ax.set_yticklabels([r[0] for r in SETTINGS])
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(tickfmt))
    ax.set_xlabel(title, color=INK)

    # value columns: x in figure-relative axes units, y in data units
    tr = ax.get_yaxis_transform()
    x0, w = rect[0], rect[2]
    cols = [(x0 + w + 0.085 - x0) / w, (x0 + w + 0.155 - x0) / w]  # right edges of the two columns
    for i, row in enumerate(SETTINGS):
        for c, k in zip(cols, (idx10, idx5)):
            ax.text(c, -i, fmt.format(row[k]), transform=tr, ha="right", va="center", color=INK,
                    clip_on=False)
    for c, colour, head in zip(cols, (GREY, BLUE), ("1/10", "1/5")):
        ax.text(c, 0.62, head, transform=tr, ha="right", va="bottom", color=INK, fontweight="bold",
                clip_on=False)
        off = (0.030 if head == "1/10" else 0.024) / w
        ax.text(c - off, 0.62, "\u25A0", transform=tr, ha="right", va="bottom", color=colour, clip_on=False)
    fig.text(x0 - 0.045, 0.80, letter, fontsize=FS_SUB, fontweight="bold", ha="left", va="bottom", color=INK)


def main():
    setup()
    fig = plt.figure(figsize=(8, 4.6), dpi=200)
    fig.patch.set_facecolor("white")
    bottom, height = 0.19, 0.57
    panel(fig, [0.06, bottom, 0.255, height], "a", "Median LDS-curve error", 1, 2, LDS_PASS, 0.075,
          [0, 0.02, 0.04, 0.06], "%.2f", "{:.3f}", undecidable=LDS_UNDECIDABLE)
    panel(fig, [0.565, bottom, 0.255, height], "b", "Median rank90 error", 3, 4, R90_PASS, 0.30,
          [0, 0.1, 0.2, 0.3], "%.1f", "{:.2f}")
    fig.text(0.015, 0.955, f"Estimator meets its criterion in all {PASS_K5} new settings with replays for 1/5 of test points",
             fontsize=FS_TITLE, fontweight="bold", ha="left", va="center", color=INK)
    fig.text(0.015, 0.902, f"{PASS_K10} of 9 with replays for 1/10 of test points. Medians over 20 draws; thresholds (dashed) set before the runs.",
             fontsize=FS_SUB, ha="left", va="center", color=MUTED)

    # colour key as one text row under the panels
    y = 0.045
    items = [("■", GREY, "replays for 1/10 of test points"), ("■", BLUE, "replays for 1/5 of test points"),
             ("□", MUTED, "H3 not judged on LDS (L* = 0.068)")]
    x = 0.015
    for sym, colour, label in items:
        t1 = fig.text(x, y, sym, color=colour, ha="left", va="center")
        t2 = fig.text(x + 0.018, y, label, color=INK, ha="left", va="center")
        fig.canvas.draw()
        x = fig.transFigure.inverted().transform(t2.get_window_extent().corners()[-1])[0] + 0.03

    out = Path(__file__).resolve().parent / "summary.png"
    fig.savefig(out, dpi=200, facecolor="white")


if __name__ == "__main__":
    main()
