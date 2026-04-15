import json
from pathlib import Path
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# 1) ALL JSON FILE PATHS
# =========================================================
JSON_FILES = [
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_a_story.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_the_white_snake.json",
    # r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_49_the_six_swans.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_96_the_three_little_birds.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_163_the_glass_coffin.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_ginger_and_pickles.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_clever_elsie.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_por_duck.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_cat-skin.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_shirtcol.json",
    # r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_storks.json",
    # r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_the_pie_and_the_patty-pan.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_daisy.json",
    # r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_the_tale_of_mrs_tiggy-winkleMANCORR.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_hansel_and_gretel.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_the_twelve_dancing_princesses.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_the_golden_bird.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_114_the_cunning_little_tailor.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_the_tale_of_peter_rabbit.json",
    r"D:\FYP\Emoi-Track\outputs\predictions\full_research_report_91_the_gnome.json",
]

# =========================================================
# 2) STANDARD SHIFT METHODS ONLY (NO NEFI HERE)
# =========================================================
STANDARD_SHIFT_METHODS = {
    "label_shift": "Label-Based",
    "local_distance_shift": "Local Distance",
    "trajectory_angle_shift": "Trajectory Angle",
    "probability_shift": "Probability-Based",
}

# =========================================================
# 3) GET SHIFT BLOCK
# =========================================================
def get_shift_block(data, method_name):
    if method_name == "probability_shift":
        return data.get("phase3", {}).get("probability_shift")
    return data.get("phase2", {}).get(method_name)

# =========================================================
# 4) PROCESS ONE STANDARD METHOD
# =========================================================
def process_method(method_key, method_label):
    class0_supports = []
    class1_supports = []
    class0_f1s = []
    class1_f1s = []

    print("\n" + "=" * 70)
    print(f"METHOD: {method_label} ({method_key})")
    print("=" * 70)

    for file_path in JSON_FILES:
        path = Path(file_path)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            shift_block = get_shift_block(data, method_key)

            if not shift_block:
                print(f"Skipping {path.name}: {method_key} not found")
                continue

            report = shift_block.get("report", {})
            class0 = report.get("0", {})
            class1 = report.get("1", {})

            c0_support = float(class0["support"])
            c1_support = float(class1["support"])
            c0_f1 = float(class0["f1-score"])
            c1_f1 = float(class1["f1-score"])

            class0_supports.append(c0_support)
            class1_supports.append(c1_support)
            class0_f1s.append(c0_f1)
            class1_f1s.append(c1_f1)

            ratio = c1_support / (c0_support + c1_support)

            print("\n----------------------------------------")
            print(f"Story: {path.name}")
            print(f"Class 0 (No Shift): Support = {c0_support:.0f}, F1 = {c0_f1:.3f}")
            print(f"Class 1 (Shift):    Support = {c1_support:.0f}, F1 = {c1_f1:.3f}")
            print(f"Shift Ratio = {ratio:.2f}")

        except Exception as e:
            print(f"Error reading {path}: {e}")

    if not class0_supports:
        raise ValueError(f"No valid files processed for {method_key}")

    total_class0 = sum(class0_supports)
    total_class1 = sum(class1_supports)
    total_samples = total_class0 + total_class1

    class0_pct = (total_class0 / total_samples) * 100
    class1_pct = (total_class1 / total_samples) * 100

    avg_class0_f1 = mean(class0_f1s)
    avg_class1_f1 = mean(class1_f1s)

    print("\n================ AGGREGATED RESULTS ================\n")
    print(f"Method: {method_label}")
    print(f"Class 0 total support: {total_class0:.0f}")
    print(f"Class 1 total support: {total_class1:.0f}")
    print(f"Class 0 (%): {class0_pct:.2f}%")
    print(f"Class 1 (%): {class1_pct:.2f}%")
    print(f"Avg Class 0 F1: {avg_class0_f1:.3f}")
    print(f"Avg Class 1 F1: {avg_class1_f1:.3f}")

    return {
        "method_key": method_key,
        "method_label": method_label,
        "total_class0": total_class0,
        "total_class1": total_class1,
        "class0_pct": class0_pct,
        "class1_pct": class1_pct,
        "avg_class0_f1": avg_class0_f1,
        "avg_class1_f1": avg_class1_f1,
    }

# =========================================================
# 5) PLOT ONE STANDARD METHOD CHART
# =========================================================
def plot_method_chart(result):
    labels = ["Class 0\n(No Shift)", "Class 1\n(Shift)"]
    support_values = [result["class0_pct"], result["class1_pct"]]
    f1_values = [result["avg_class0_f1"] * 100, result["avg_class1_f1"] * 100]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    bars1 = ax.bar(x - width / 2, support_values, width, label="Support (%)", color="#4C78A8")
    bars2 = ax.bar(x + width / 2, f1_values, width, label="Average F1 (%)", color="#F58518")

    ax.set_title(f"Class Distribution and Average F1 for {result['method_label']}", fontsize=13)
    ax.set_ylabel("Percentage")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend()

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{bar.get_height():.1f}%",
            ha="center"
        )

    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{bar.get_height():.1f}%",
            ha="center"
        )

    plt.tight_layout()
    output_name = f"{result['method_key']}_aggregated_dual_bar_chart.png"
    plt.savefig(output_name, dpi=300)
    plt.show()
    print(f"\nChart saved as: {output_name}")

# =========================================================
# 6) PLOT STANDARD METHOD SUMMARY CHART
# =========================================================
def plot_summary_chart(results):
    method_labels = [r["method_label"] for r in results]
    shift_f1_values = [r["avg_class1_f1"] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(method_labels, shift_f1_values, color="#59A14F")

    ax.set_title("Average Shift-Class F1 Across Standard Shift Detection Methods", fontsize=13)
    ax.set_ylabel("Average Class 1 F1 (%)")
    ax.set_ylim(0, 70)
    plt.xticks(rotation=20, ha="right")

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{bar.get_height():.1f}%",
            ha="center"
        )

    plt.tight_layout()
    output_name = "summary_standard_shift_method_comparison.png"
    plt.savefig(output_name, dpi=300)
    plt.show()
    print(f"\nSummary chart saved as: {output_name}")

# =========================================================
# 7) SEPARATE NEFI TABLE
# =========================================================
def process_nefi_table():
    rows = []
    nefi_rupture_f1s = []
    top5_precision_list = []
    top5_matches_list = []
    top5_f1_list = []

    print("\n" + "=" * 110)
    print("TABLE 2: NEFI EVALUATION")
    print("=" * 110)
    print(
        f"{'Story':<45}"
        f"{'NEFI F1':<12}"
        f"{'Top5 P@5':<12}"
        f"{'Matches':<12}"
        f"{'Top5 F1':<12}"
    )

    for file_path in JSON_FILES:
        path = Path(file_path)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            phase2 = data.get("phase2", {})
            nefi_block = phase2.get("nefi_rupture_shift", {})
            top5_block = phase2.get("top5_nefi_shift", {})

            if not nefi_block or not top5_block:
                print(f"Skipping {path.name}: missing NEFI blocks")
                continue

            nefi_f1 = float(nefi_block.get("f1", 0))
            top5_p5 = float(top5_block.get("precision_at_5", 0))
            top5_matches = int(top5_block.get("matches_top5", 0))
            top5_f1 = float(top5_block.get("f1", 0))

            story_name = path.stem.replace("full_research_report_", "")

            rows.append((story_name, nefi_f1, top5_p5, top5_matches, top5_f1))

            nefi_rupture_f1s.append(nefi_f1)
            top5_precision_list.append(top5_p5)
            top5_matches_list.append(top5_matches)
            top5_f1_list.append(top5_f1)

            print(
                f"{story_name:<45}"
                f"{nefi_f1:<12.3f}"
                f"{top5_p5:<12.2f}"
                f"{str(top5_matches) + '/5':<12}"
                f"{top5_f1:<12.3f}"
            )

        except Exception as e:
            print(f"Error reading {path}: {e}")

    if not rows:
        raise ValueError("No valid NEFI rows found.")

    avg_nefi_f1 = mean(nefi_rupture_f1s)
    avg_top5_p5 = mean(top5_precision_list)
    avg_top5_matches = mean(top5_matches_list)
    avg_top5_f1 = mean(top5_f1_list)

    print("-" * 110)
    print(
        f"{'AVERAGE':<45}"
        f"{avg_nefi_f1:<12.3f}"
        f"{avg_top5_p5:<12.2f}"
        f"{(str(round(avg_top5_matches, 2)) + '/5'):<12}"
        f"{avg_top5_f1:<12.3f}"
    )

    return {
        "avg_nefi_f1": avg_nefi_f1,
        "avg_top5_precision_at_5": avg_top5_p5,
        "avg_top5_matches": avg_top5_matches,
        "avg_top5_f1": avg_top5_f1,
        "rows": rows,
    }

# =========================================================
# 8) RUN STANDARD METHODS
# =========================================================
all_results = []

for method_key, method_label in STANDARD_SHIFT_METHODS.items():
    result = process_method(method_key, method_label)
    all_results.append(result)
    plot_method_chart(result)

# =========================================================
# 9) TABLE 1 SUMMARY IN TERMINAL
# =========================================================
print("\n" + "=" * 90)
print("TABLE 1: FINAL SUMMARY ACROSS STANDARD SHIFT METHODS")
print("=" * 90)
print(f"{'Method':<22}{'Class0 %':<12}{'Class1 %':<12}{'Avg C0 F1':<14}{'Avg C1 F1':<14}")
for r in all_results:
    print(
        f"{r['method_label']:<22}"
        f"{r['class0_pct']:<12.2f}"
        f"{r['class1_pct']:<12.2f}"
        f"{r['avg_class0_f1']:<14.3f}"
        f"{r['avg_class1_f1']:<14.3f}"
    )

# =========================================================
# 10) STANDARD SUMMARY CHART
# =========================================================
plot_summary_chart(all_results)

# =========================================================
# 11) TABLE 2: NEFI EVALUATION
# =========================================================
nefi_results = process_nefi_table()

print("\n" + "=" * 70)
print("NEFI AGGREGATED SUMMARY")
print("=" * 70)
print(f"Average NEFI Rupture F1: {nefi_results['avg_nefi_f1']:.3f}")
print(f"Average Top-5 Precision@5: {nefi_results['avg_top5_precision_at_5']:.3f}")
print(f"Average Top-5 Matches: {nefi_results['avg_top5_matches']:.2f}/5")
print(f"Average Top-5 F1: {nefi_results['avg_top5_f1']:.3f}")