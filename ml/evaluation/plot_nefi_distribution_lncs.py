import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# LOAD STORY DATA
# ==========================================

file_path = r"outputs/predictions/EmoiTrack_T0.88_the_roly-poly_pudding_ORIGINAL.csv"
df = pd.read_csv(file_path)

nefi_scores = df["NEFI"].values
rupture_flags = df["Rupture_Flag"].values

threshold = nefi_scores.mean() + nefi_scores.std()

# Separate rupture vs non-rupture
rupture_values = nefi_scores[rupture_flags == 1]
non_rupture_values = nefi_scores[rupture_flags == 0]

# ==========================================
# LNCS Formatting
# ==========================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300
})

fig, ax = plt.subplots(figsize=(6.2, 4))

# ---- Base Histogram (all values) ----
counts, bins, patches = ax.hist(
    nefi_scores,
    bins=20,
    color="lightgray",
    edgecolor="black",
    linewidth=0.6,
    label="All NEFI"
)

# ---- Overlay rupture values as red ticks (rug style) ----
ax.scatter(
    rupture_values,
    np.zeros_like(rupture_values) - 0.5,  # slightly below x-axis
    marker="o",
    color="black",
    s=20,
    label="Rupture Points",
    zorder=5
)

# ---- Threshold Line ----
ax.axvline(
    threshold,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="Rupture Threshold ($\\mu + \\sigma$)"
)

ax.set_xlabel("NEFI Score")
ax.set_ylabel("Frequency")

ax.legend(frameon=False)

plt.tight_layout()
plt.savefig("nefi_distribution_roly_poly.pdf", format="pdf", bbox_inches="tight")
plt.close()

print("NEFI distribution with rupture highlights saved.")