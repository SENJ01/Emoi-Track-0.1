import matplotlib.pyplot as plt
import numpy as np

# Data (aggregated mean across 5 stories)
detectors = [
    "Label Shift",
    "Local Distance",
    "Trajectory Angle",
    "NEFI Rupture"
]

macro_f1 = [0.590, 0.571, 0.400, 0.551]

# Optional: standard deviations (if you computed them)
# std_dev = [0.045, 0.038, 0.060, 0.052]

# Use serif font for LNCS compatibility
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9
})

fig, ax = plt.subplots(figsize=(4.5, 3))  # Compact LNCS-friendly size

x = np.arange(len(detectors))

bars = ax.bar(
    x,
    macro_f1,
    color="white",
    edgecolor="black",
    linewidth=1.0
)

# If using error bars (recommended)
# ax.errorbar(
#     x,
#     macro_f1,
#     yerr=std_dev,
#     fmt='none',
#     ecolor='black',
#     capsize=3,
#     linewidth=1
# )

ax.set_xticks(x)
ax.set_xticklabels(detectors, rotation=30, ha="right")
ax.set_ylabel("Macro-F1")
ax.set_ylim(0, 1.0)

ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

plt.tight_layout()

# Save as vector PDF for Springer
plt.savefig("detector_macro_f1.pdf", format="pdf", bbox_inches="tight")
plt.show()