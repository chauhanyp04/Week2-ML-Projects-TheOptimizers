import numpy as np
import matplotlib.pyplot as plt

# ── Dataset Creation ──────────────────────────────────────────────────────────
# We create a simple dataset: Hours Studied vs Exam Score
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
y = np.array([52, 58, 63, 70, 75, 80, 85, 88, 92, 97], dtype=float)

# ── Step 1: Calculate Slope (m) ───────────────────────────────────────────────
n = len(X)
x_mean = np.mean(X)
y_mean = np.mean(y)

numerator   = np.sum((X - x_mean) * (y - y_mean))
denominator = np.sum((X - x_mean) ** 2)
slope = numerator / denominator

# ── Step 2: Calculate Intercept (b) ──────────────────────────────────────────
intercept = y_mean - slope * x_mean

# ── Step 3: Prediction ────────────────────────────────────────────────────────
y_pred = slope * X + intercept

# ── Step 4: Mean Squared Error ────────────────────────────────────────────────
mse = np.mean((y - y_pred) ** 2)

# ── Output ────────────────────────────────────────────────────────────────────
print("=" * 45)
print("   LINEAR REGRESSION FROM SCRATCH")
print("=" * 45)
print(f"Slope (m)         : {slope:.4f}")
print(f"Intercept (b)     : {intercept:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print()
print(f"{'Hours':>8}  {'Actual':>8}  {'Predicted':>10}")
print("-" * 30)
for h, actual, pred in zip(X, y, y_pred):
    print(f"{h:>8.0f}  {actual:>8.1f}  {pred:>10.2f}")

# ── Step 5: Visualization ─────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='steelblue', s=80, zorder=5, label='Actual Data')
plt.plot(X, y_pred, color='crimson', linewidth=2, label=f'Regression Line\ny = {slope:.2f}x + {intercept:.2f}')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Linear Regression From Scratch\n(Hours Studied vs Exam Score)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/lr_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved!")
