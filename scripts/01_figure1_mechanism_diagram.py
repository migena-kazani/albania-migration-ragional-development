# -*- coding: utf-8 -*-
"""
Script 01: Figure 1 - Cumulative Spatial Migration Mechanism (conceptual diagram)

Produces:
  - Figure 1. Cumulative Spatial Migration Mechanism: The Three-Tier Hierarchy and
    the Core Feedback Loop

IMPORTANT - this script is different from every other script in this package: it
draws a conceptual/illustrative schematic, not a plot of empirical results. It is
not derived from any regression or data file, and does not depend on any other
script running first. It visualises, for the reader, the theoretical mechanism the
paper's dynamic-panel GMM persistence coefficient (gamma=0.4525, Table 14) is
interpreted as the empirical counterpart of:

  - Three concentric tiers (Periphery -> Coastal/Transitional Buffer -> Core/Tirana),
    matching the k-means/hierarchical clustering result (Tables 23-25, Figures 4-5).
  - Dashed inward arrows: net migration flowing from the periphery and the coastal
    buffer toward the core.
  - A solid internal loop: the within-core cumulative-causation cycle (net
    in-migration -> labour supply & market size -> agglomeration & investment ->
    GDP/labour demand -> net in-migration), in the spirit of Myrdal and Krugman.

Because this is an original schematic rather than a reproduction of copyrighted
material, feel free to adjust labels/wording to taste; the geometry below matches
the figure as it appears in the manuscript.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(10, 8.6))
ax.set_xlim(-10, 10)
ax.set_ylim(-9.5, 9.5)
ax.set_aspect("equal")
ax.axis("off")

# ---- Three concentric tiers ---------------------------------------------------
periphery = Circle((0, 0), 8.6, facecolor="#f2f2f2", edgecolor="#555555", linewidth=1.3, zorder=1)
buffer_ring = Circle((0, 0), 5.6, facecolor="#d9d9d9", edgecolor="#555555", linewidth=1.3, zorder=2)
core = Circle((0, 0), 2.6, facecolor="#8c8c8c", edgecolor="#333333", linewidth=1.3, zorder=3)
for c in (periphery, buffer_ring, core):
    ax.add_patch(c)

ax.text(0, 7.9, "PERIPHERY", ha="center", va="center", fontsize=13)
ax.text(0, 7.35, "(remaining regions: continuous population loss)", ha="center", va="center", fontsize=10.5)

ax.text(0, 5.45, "COASTAL / TRANSITIONAL BUFFER (Durrës, Vlorë, Lezhë)",
        ha="center", va="center", fontsize=11)

ax.text(0, 0, "CORE\n(Tirana)", ha="center", va="center", fontsize=14, fontweight="bold", color="white")

# ---- Dashed inward arrows: net migration, periphery/buffer -> core ------------
for angle_deg in (135, 45, 225, -45):
    a = np.radians(angle_deg)
    x0, y0 = 7.2 * np.cos(a), 7.2 * np.sin(a)
    x1, y1 = 2.9 * np.cos(a), 2.9 * np.sin(a)
    arrow = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                             linestyle=(0, (5, 4)), linewidth=1.6, color="#b03a2e", zorder=4)
    ax.add_patch(arrow)

# ---- Solid internal loop: within-core cumulative-causation cycle --------------
loop_r = 3.9
loop_stages = [
    (90,  "Net in-migration ↑"),
    (0,   "GDP / labour\ndemand ↑"),
    (-90, "Agglomeration &\ninvestment ↑"),
    (180, "Labour supply &\nmarket size ↑"),
]
label_r = 4.75
for angle_deg, label in loop_stages:
    a = np.radians(angle_deg)
    ax.text(label_r * np.cos(a), label_r * np.sin(a), label, ha="center", va="center",
            fontsize=10.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", linewidth=0.8))

n_pts = len(loop_stages)
for i in range(n_pts):
    a0 = np.radians(loop_stages[i][0] - 18)
    a1 = np.radians(loop_stages[(i + 1) % n_pts][0] + 18 - 360 * (i == n_pts - 1))
    thetas = np.linspace(np.radians(loop_stages[i][0]) - np.radians(5),
                          np.radians(loop_stages[i][0]) - np.radians(85), 30)
    xs = loop_r * np.cos(thetas)
    ys = loop_r * np.sin(thetas)
    ax.plot(xs, ys, color="black", linewidth=1.4, zorder=4)
    arrow = FancyArrowPatch((xs[-2], ys[-2]), (xs[-1], ys[-1]), arrowstyle="-|>",
                             mutation_scale=14, linewidth=1.4, color="black", zorder=4)
    ax.add_patch(arrow)

ax.text(0, -9.1,
        "Dashed arrows: net migration flow (periphery/coastal → core). "
        "Solid loop: within-core cumulative-causation cycle.",
        ha="center", va="center", fontsize=10.5)

plt.tight_layout()
plt.savefig("../figures/figure1_mechanism_diagram.png", dpi=200, bbox_inches="tight")
print("Figure 1 (conceptual mechanism diagram) saved to ../figures/figure1_mechanism_diagram.png")
