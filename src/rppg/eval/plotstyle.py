"""Shared figure style for report figures (static PNGs, light background).

Categorical colours are taken in fixed order from a palette validated for
colour-vision deficiency (adjacent pairs); three of the slots are below 3:1
contrast on white, so bar figures carry value labels and every figure has a
table alongside it in the report.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def apply() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "font.size": 9,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
        }
    )


def grouped_bars(ax, groups, series, values, colors=None, fmt="{:.1f}"):
    """Grouped vertical bars with a 2 px gap between bars and value labels.

    values[s][g] is the value of series s in group g (NaN = no bar).
    """
    import numpy as np

    colors = colors or SERIES
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, name in enumerate(series):
        v = np.asarray(values[i], dtype=float)
        pos = x - 0.4 + width * (i + 0.5)
        bars = ax.bar(pos, np.nan_to_num(v), width * 0.92, color=colors[i], label=name, edgecolor=SURFACE, linewidth=1)
        for b, val in zip(bars, v):
            if np.isnan(val):
                b.set_visible(False)
                continue
            below = val < 0  # label sits beyond the bar end, on whichever side it points
            ax.annotate(fmt.format(val), (b.get_x() + b.get_width() / 2, val), xytext=(0, -2 if below else 2),
                        textcoords="offset points", ha="center", va="top" if below else "bottom",
                        fontsize=6.5, color=TEXT_SECONDARY)
    ax.set_xticks(x, groups)
    ax.tick_params(axis="x", length=0)


def legend_below(fig, ax, ncols: int, title: str | None = None) -> None:
    """One legend for the whole figure, in a row below the panels."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower left", ncols=ncols, title=title, alignment="left")
