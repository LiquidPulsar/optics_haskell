import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def parse(path, cutoff=None):
    epochs, accs = [], []
    with open(path) as f:
        for line in f:
            m = re.search(r'epoch (\d+) accuracy ([0-9.e+-]+)', line)
            if m:
                e, a = int(m.group(1)), float(m.group(2)) * 100
                if cutoff is not None and e > cutoff:
                    break
                epochs.append(e)
                accs.append(a)
    return epochs, accs

CUTOFF = 300

epochs, accs = parse("mnist_resid_train_data.txt", cutoff=CUTOFF)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(epochs, accs, linewidth=1.5, color="#4C72B0")

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Residual MNIST Training Accuracy", fontsize=13)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_ylim(0, 100)
ax.set_xlim(0, CUTOFF)

fig.tight_layout()
fig.savefig("mnist_resid_training.pdf", dpi=150)
fig.savefig("mnist_resid_training.png", dpi=150)
print(f"Saved (epochs 0–{CUTOFF})")
print(f"Peak in range: {max(accs):.2f}% at epoch {epochs[accs.index(max(accs))]}")
