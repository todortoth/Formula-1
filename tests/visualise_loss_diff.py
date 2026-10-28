import re

import matplotlib.pyplot as plt

# Feltételezzük, hogy a logok 'log1.txt', 'log2.txt' és 'log3.txt' néven vannak elmentve
log_files = {
    "Flatten": "runs/logs/pre3_short_flatten.log",
    "Sector": "runs/logs/pre3_short_sector.log",
    "GNN": "runs/logs/pre3_short_gnn.log",
}

plt.figure(figsize=(11, 6))

# Színek a különböző logok elkülönítéséhez
colors = {"Flatten": "blue", "Sector": "green", "GNN": "red"}

for label, filename in log_files.items():
    steps, losses = [], []
    val_steps, val_mses = [], []

    try:
        with open(filename, encoding="utf-8") as f:
            for line in f:
                # Képzési veszteség (recon loss) kinyerése
                train_match = re.search(r"step\s+(\d+)\s+recon\s+loss\s+([\d.]+)", line)
                if train_match:
                    steps.append(int(train_match.group(1)))
                    losses.append(float(train_match.group(2)))

                # Validációs hiba (mean MSE) kinyerése
                val_match = re.search(r"mean MSE over BTPZ:\s+([\d.]+)", line)
                if val_match and steps:
                    # A legutolsó ismert lépést rendeljük a validációs ponthoz
                    val_steps.append(steps[-1])
                    val_mses.append(float(val_match.group(1)))

        # Képzési veszteség kirajzolása szaggatott vonallal
        plt.plot(steps, losses, linestyle="--", alpha=0.4, color=colors[label], label=f"{label} - Train Loss")

        # Validációs MSE kirajzolása folytonos vonallal és pontokkal
        plt.plot(val_steps, val_mses, marker="o", linewidth=2, color=colors[label], label=f"{label} - Val Mean MSE")

    except FileNotFoundError:
        print(f"Hiba: A(z) {filename} nem található. Kérlek hozd létre a script mellett.")

# Diagram formázása
plt.title("Tanítási folyamat összehasonlítása (Reconstruction Loss és Validation MSE)")
plt.xlabel("Lépések (Steps)")
plt.ylabel("Érték")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()

# Diagram elmentése képként
plt.savefig("loss_comparison_plot.png", dpi=300)
print("A diagram sikeresen elmentve 'loss_comparison_plot.png' néven!")
# plt.show() # Ha lokálisan futtatod, ezt kikommentelheted a megjelenítéshez
