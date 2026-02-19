import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("results/depth_comparison.csv")

plt.figure(figsize=(6,4))

plt.plot(df["max_depth"], df["accuracy"], marker="o", label="Accuracy")
plt.plot(df["max_depth"], df["female_DI"], marker="o", label="Female DI")

plt.xlabel("Max Tree Depth")
plt.ylabel("Metric Value")
plt.title("Fairness–Performance Trade-off")
plt.grid(True)
plt.legend()

Path("results/figures").mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig("results/figures/depth_tradeoff.png", dpi=300)
plt.show()
