import ast
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

with open("iris_train_data.txt") as f:
    raw = f.read().strip()

CUTOFF = 100
pairs = ast.literal_eval(raw)[:CUTOFF + 1]
epochs = list(range(len(pairs)))
accs   = [a * 100 for _, a in pairs]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(epochs, accs, linewidth=1.5, color="#4C72B0")

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Iris Training Accuracy", fontsize=13)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_ylim(0, 100)
ax.set_xlim(0, CUTOFF)

fig.tight_layout()
fig.savefig("iris_training.pdf", dpi=150)
fig.savefig("iris_training.png", dpi=150)

peak = max(accs)
peak_ep = accs.index(peak)
print(f"Total epochs: {len(epochs)}")
print(f"Peak: {peak:.2f}% at epoch {peak_ep}")
print(f"Epoch 0: {accs[0]:.2f}%")
print(f"Epoch 50: {accs[50]:.2f}%" if len(accs) > 50 else "")
print(f"Epoch 100: {accs[100]:.2f}%" if len(accs) > 100 else "")
