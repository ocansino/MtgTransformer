# plot_model_comparison.py

# plot_model_comparison.py

import matplotlib.pyplot as plt
import numpy as np

models = ["Global", "Position", "Bigram", "Transformer"]

# Top-1 accuracies (all models)
top1 = [0.0072, 0.0219, 0.0289, 0.0425]

# Top-5 only for Transformer (others set to 0 for plotting)
top5 = [0, 0, 0, 0.1900]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(9, 5))

# Top-1 bars
bars1 = plt.bar(x - width/2, top1, width, label="Top-1 Accuracy")

# Top-5 bars (only meaningful for Transformer)
bars2 = plt.bar(x + width/2, top5, width, label="Top-5 Accuracy")

plt.xticks(x, models)
plt.ylabel("Accuracy")
plt.title("Model Performance Comparison")

# Add value labels
for bar in bars1:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.4f}",
        ha="center",
        va="bottom",
    )

for i, bar in enumerate(bars2):
    height = bar.get_height()
    if height > 0:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )
    else:
        # Label that top-5 not available
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            0.001,
            "N/A",
            ha="center",
            va="bottom",
            fontsize=8,
            color="gray",
        )

plt.ylim(0, 0.21)
plt.legend()
plt.tight_layout()

plt.savefig("model_comparison.png", dpi=300)
plt.show()