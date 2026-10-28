import json
import re

import matplotlib.pyplot as plt

# A log fájlok elérési útja (.jsonl formátumra frissítve)
log_files = {
    "Linear": "runs/ft_m_lin/log.jsonl",
    "MLP": "runs/ft_m_mlp/log.jsonl",
    "CAR": "runs/ft_m_car/log.jsonl",
}

plt.figure(figsize=(11, 6))

# Színek a különböző logok elkülönítéséhez
colors = {"Linear": "blue", "MLP": "green", "CAR": "red"}

for label, filename in log_files.items():
    steps, losses = [], []
    val_steps, val_mses = [], []

    try:
        with open(filename, encoding="utf-8") as f:
            is_jsonl = filename.endswith(".jsonl")

            for line in f:
                line = line.strip()
                if not line:
                    continue

                if is_jsonl:
                    # --- JSONL feldolgozás ---
                    try:
                        data = json.loads(line)
                        # A beillesztett mintád alapján a 'step' és 'loss' kulcsokat keressük
                        if "step" in data and "loss" in data:
                            steps.append(int(data["step"]))
                            losses.append(float(data["loss"]))

                        # Ha van külön validációs kulcs (pl. val_loss vagy mean_mse), azt itt kezelheted:
                        if "mean_mse" in data and steps:
                            val_steps.append(steps[-1])
                            val_mses.append(float(data["mean_mse"]))
                    except json.JSONDecodeError:
                        continue  # Hibás sor átugrása
                else:
                    # --- Hagyományos .log (Regex) feldolgozás ---
                    train_match = re.search(r"step\s+(\d+)\s+recon\s+loss\s+([\d.]+)", line)
                    if train_match:
                        steps.append(int(train_match.group(1)))
                        losses.append(float(train_match.group(2)))

                    val_match = re.search(r"mean MSE over BTPZ:\s+([\d.]+)", line)
                    if val_match and steps:
                        val_steps.append(steps[-1])
                        val_mses.append(float(val_match.group(1)))

        # Képzési veszteség kirajzolása (ha van adat)
        if steps:
            plt.plot(steps, losses, linestyle="--", alpha=0.4, color=colors[label], label=f"{label} - Train Loss")

        # Validációs hiba kirajzolása (ha van adat)
        if val_steps:
            plt.plot(val_steps, val_mses, marker="o", linewidth=2, color=colors[label], label=f"{label} - Val MSE")

    except FileNotFoundError:
        print(f"Hiba: A(z) {filename} nem található. Kérlek ellenőrizd az elérési utat.")

# Diagram formázása
plt.title("Tanítási folyamat összehasonlítása (Loss)")
plt.xlabel("Lépések (Steps)")
plt.ylabel("Érték")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()

# Diagram elmentése képként
plt.savefig("loss_comparison_plot.png", dpi=300)
print("A diagram sikeresen elmentve 'loss_comparison_plot.png' néven!")
