# plot_accuracy_by_position.py

import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "accuracy_by_position.csv"

df = pd.read_csv(CSV_PATH)

plt.figure(figsize=(10, 5))

plt.plot(df["position"], df["top1_accuracy"], marker="o", label="Top-1 Accuracy")
plt.plot(df["position"], df["top5_accuracy"], marker="o", label="Top-5 Accuracy")

# Mark pack boundaries
plt.axvline(x=15, linestyle="--", alpha=0.5)
plt.axvline(x=30, linestyle="--", alpha=0.5)

plt.xlabel("Prediction Position")
plt.ylabel("Accuracy")
plt.title("Transformer Accuracy by Draft Position")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("accuracy_by_position.png", dpi=300)
plt.show()