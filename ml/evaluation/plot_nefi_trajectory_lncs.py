import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# LOAD STORY DATA (Roly-Poly Pudding)
# ==========================================

file_path = r"outputs/predictions/EmoiTrack_T0.88_the_roly-poly_pudding_ORIGINAL.csv"

df = pd.read_csv(file_path)

emotion_labels = ["Score_0", "Score_1", "Score_2"]
emotion_probs = df[emotion_labels].values

nefi_scores = df["NEFI"].values
rupture_flags = df["Rupture_Flag"].values
# ==========================================
# LNCS / Springer Formatting
# ==========================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300
})

N = len(nefi_scores)
x = np.arange(N)

fig, ax1 = plt.subplots(figsize=(12, 5)) 

# ---- Emotion Trajectoriess---
line_styles = ['-', '--', '-.']

for i in range(emotion_probs.shape[1]):
    ax1.plot(
        x,
        emotion_probs[:, i],
        linestyle=line_styles[i % len(line_styles)],
        linewidth=0.8,      # thinner
        alpha=0.7,          # slightly transparent
        label=emotion_labels[i]
    )

ax1.set_xlabel("Sentence Index")
ax1.set_ylabel("Emotion Probability")
ax1.set_xlim(0, N - 1)
ax1.set_ylim(0, 1)

# ---- Show ticks every 5 sentences ----
ax1.set_xticks(np.arange(0, N, 10))

# ---- NEFI Axis ----
ax2 = ax1.twinx()

ax2.plot(
    x,
    nefi_scores,
    color="black",
    linewidth=1,     # slightly stronger than emotion lines
    linestyle="-",
    alpha=0.9,
    label="NEFI"
)

ax2.set_ylabel("NEFI Score")

# ---- Rupture Points (smaller markers) ----
rupture_indices = np.where(rupture_flags == 1)[0]

ax2.scatter(
    rupture_indices,
    nefi_scores[rupture_indices],
    color="black",
    marker="o",
    s=12,              # smaller
    zorder=5,
    label="Rupture"
)

# ---- Merge Legends (compact layout) ----
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper right",
    frameon=False,
    ncol=2             # more compact legend
)

plt.tight_layout()
plt.savefig("nefi_trajectory_roly_poly.pdf", format="pdf", bbox_inches="tight")
plt.close()

print("Clean figure saved as nefi_trajectory_roly_poly.pdf")

print("Figure saved as nefi_trajectory_roly_poly.pdf")