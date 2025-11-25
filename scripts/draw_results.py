import matplotlib.pyplot as plt
import numpy as np
import os

# Data for Plot 1
countries = ["US", "DE", "FR"]
methods = ["Zero-shot", "Zero-shot (chat)", "Many-shot", "Gold-label"]

data_p1 = {
    "US": [0.5143, 0.6393, 0.8000, 0.7714],
    "DE": [0.6230, 0.6230, 0.7429, 0.6429],
    "FR": [0.6167, 0.6167, 0.8143, 0.8000],
}

# Calculate averages
averages = [np.mean([data_p1[c][i] for c in countries]) for i in range(len(methods))]

# Data for Plot 2 (US only)
num_examples = [0, 10, 20, 30, 40, 50]
data_p2 = {
    "Many-shot": [0.5143, 0.7714, 0.7714, 0.8286, 0.8286, 0.8000],
    "Gold-label": [0.5143,0.7143, 0.7143, 0.7429, 0.7857, 0.7714],
    # "Random (0.5 acc)": [0.5143, 0.6571, 0.6571, 0.6143, 0.6714, 0.7714],
    "Random (0.8 acc)": [0.5143, 0.7286, 0.7000, 0.6286, 0.6857, 0.6143],
}


def draw_plots():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Average Accuracy Comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        methods, averages, alpha=0.8
    )

    plt.title("Average Accuracy Comparison (US, DE, FR)", fontsize=14)
    plt.ylabel("Average Accuracy", fontsize=12)
    plt.ylim(0, 1.0)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()
    save_path1 = os.path.join(output_dir, "average_accuracy_comparison.png")
    plt.savefig(save_path1, dpi=300)
    print(f"Saved Plot 1 to {save_path1}")
    plt.close()

    # Plot 2: Accuracy vs Number of Examples
    plt.figure(figsize=(10, 6))

    markers = ["o", "s", "^", "D"]

    for i, (method, values) in enumerate(data_p2.items()):
        plt.plot(
            num_examples,
            values,
            marker=markers[i],
            label=method,
            linewidth=2,
            markersize=8,
        )

    plt.title("Accuracy vs Number of In-Context Examples (US)", fontsize=14)
    plt.xlabel("Number of In-Context Examples", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0.5, 0.9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)

    plt.tight_layout()
    save_path2 = os.path.join(output_dir, "accuracy_vs_examples.png")
    plt.savefig(save_path2, dpi=300)
    print(f"Saved Plot 2 to {save_path2}")
    plt.close()


if __name__ == "__main__":
    draw_plots()
