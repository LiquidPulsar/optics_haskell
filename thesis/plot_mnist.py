import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def parse(path):
    epochs, accs = [], []
    with open(path) as f:
        for line in f:
            m = re.search(r'epoch (\d+) accuracy ([0-9.e+-]+)', line)
            if m:
                epochs.append(int(m.group(1)))
                accs.append(float(m.group(2)) * 100)
    return epochs, accs

epochs, accs = parse("mnist_train_data_2.txt")

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(epochs, accs, linewidth=1.5, color="#4C72B0")

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("MNIST Training Accuracy", fontsize=13)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_ylim(0, 100)

fig.tight_layout()
fig.savefig("mnist_training.pdf", dpi=150)
fig.savefig("mnist_training.png", dpi=150)
print("Saved mnist_training.pdf and mnist_training.png")
