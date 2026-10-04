"""Step 1.9 — Side-by-side figure of target vs user skeletons.

Both skeletons are drawn in normalized coordinates (hips at 0, torso length 1)
on the same scale, with the y-axis inverted so people appear upright.
User joints are coloured by how well they match, with a different marker shape
per rating so the result doesn't rely on colour alone.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from dance_challenge.config import BONES, GOOD_SCORE, OKAY_SCORE
from dance_challenge.feedback import pose_feedback
from dance_challenge.scoring import ScoreResult

# (colour, marker, label) per rating.
GOOD_STYLE = ("#0ca30c", "o", f"good (≥{GOOD_SCORE:.0f})")
OKAY_STYLE = ("#fab219", "^", f"okay ({OKAY_SCORE:.0f}–{GOOD_SCORE - 1:.0f})")
POOR_STYLE = ("#d03b3b", "X", f"poor (<{OKAY_SCORE:.0f})")
SKIPPED_STYLE = ("#9a9a9a", "o", "not visible")
TARGET_COLOR = "#3b6fb6"
BONE_COLOR = "#606060"


def joint_style(score: float | None) -> tuple[str, str, str]:
    """(colour, marker, label) for a joint score (None = skipped)."""
    if score is None:
        return SKIPPED_STYLE
    if score >= GOOD_SCORE:
        return GOOD_STYLE
    if score >= OKAY_SCORE:
        return OKAY_STYLE
    return POOR_STYLE


def _draw_skeleton(ax, points: dict[str, np.ndarray], joint_colors=None) -> None:
    """Draw bones and joints. ``joint_colors`` maps joint -> (colour, marker)."""
    for a, b in BONES:
        if a in points and b in points:
            ax.plot([points[a][0], points[b][0]], [points[a][1], points[b][1]],
                    color=BONE_COLOR, linewidth=2, solid_capstyle="round", zorder=1)
    for joint, (x, y) in points.items():
        color, marker = (joint_colors or {}).get(joint, (TARGET_COLOR, "o"))
        size = 110 if joint == "nose" else 80
        ax.scatter([x], [y], s=size, c=color, marker=marker,
                   edgecolors="white", linewidths=1.5, zorder=2)


def _shared_limits(*point_sets: dict[str, np.ndarray]) -> tuple[tuple, tuple]:
    """Square axis limits that fit every skeleton, with a margin."""
    allpts = np.array([p for pts in point_sets for p in pts.values()])
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    centre, half = (lo + hi) / 2, max(hi - lo) / 2 + 0.4
    return (centre[0] - half, centre[0] + half), (centre[1] - half, centre[1] + half)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}"


def plot_comparison(
    result: ScoreResult,
    target_name: str,
    user_label: str,
    out_path: str | Path,
) -> Path:
    """Save the comparison figure and return its path.

    Requires a visible result (``result.visible`` is True).
    """
    if not result.visible:
        raise ValueError("Cannot plot a pose that is not fully visible")

    target_pts, user_pts = result.target_norm, result.user_norm
    xlim, ylim = _shared_limits(target_pts, user_pts)

    fig = plt.figure(figsize=(10, 8))
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1.1], hspace=0.02, wspace=0.05)
    ax_t, ax_u = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ax_text = fig.add_subplot(grid[1, :])

    _draw_skeleton(ax_t, target_pts)
    user_colors = {}
    for joint in user_pts:
        color, marker, _ = joint_style(result.joint_scores.get(joint))
        user_colors[joint] = (color, marker)
    _draw_skeleton(ax_u, user_pts, user_colors)

    for ax, title in ((ax_t, f"Target: {target_name}"), (ax_u, f"You: {user_label}")):
        ax.set_xlim(*xlim)
        ax.set_ylim(ylim[1], ylim[0])  # inverted: image y points down
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#d0d0d0")

    legend = [Line2D([], [], linestyle="", marker=m, markersize=9, markerfacecolor=c,
                     markeredgecolor="white", color=c, label=lbl)
              for c, m, lbl in (GOOD_STYLE, OKAY_STYLE, POOR_STYLE, SKIPPED_STYLE)]
    ax_text.legend(handles=legend, loc="upper right", fontsize=9.5, frameon=False,
                   title="Your joints", title_fontsize=10, alignment="left")

    # Score summary and feedback below the figures.
    ax_text.axis("off")
    header = (f"Score: {result.overall}/100     "
              f"Upper body: {_fmt(result.upper_body)}     "
              f"Lower body: {_fmt(result.lower_body)}")
    ax_text.text(0.0, 1.0, header, fontsize=15, fontweight="bold", va="top",
                 transform=ax_text.transAxes)
    lines = [fb.text() for fb in pose_feedback(result)]
    if result.skipped_joints:
        lines.append("Skipped (low visibility): " + ", ".join(result.skipped_joints))
    ax_text.text(0.0, 0.72, "\n".join(lines), fontsize=11.5, va="top",
                 linespacing=1.5, transform=ax_text.transAxes)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
