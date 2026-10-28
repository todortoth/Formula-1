import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path):
    train_steps, train_loss = [], []
    val_steps = []
    val_mean, val_short, val_med, val_long = [], [], [], []

    # Task-specifikus MSE-k tárolói (pl. short eval alapján vagy átlagolva)
    task_steps = []
    task_B, task_T, task_P, task_Z = [], [], [], []

    current_step = 0
    print(f"Szöveges log feldolgozása: {log_path}")

    train_pattern = re.compile(r"step\s+(\d+)\s+recon\s+loss\s+([\d\.]+)")
    val_pattern = re.compile(
        r"\[val\]\s+mean MSE over BTPZ:\s+([\d\.]+).*?\[short\s+([\d\.]+),\s+medium\s+([\d\.]+),\s+long\s+([\d\.]+)\]"
    )
    eval_pattern = re.compile(r"\[eval\s+(short|medium|long)\]\s+\{(.*?)\}")

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            # 1. Train lépés
            train_match = train_pattern.search(line)
            if train_match:
                current_step = int(train_match.group(1))
                train_steps.append(current_step)
                train_loss.append(float(train_match.group(2)))
                continue

            # 2. Validációs összesítő (itt mentjük a lépést is a validációhoz)
            val_match = val_pattern.search(line)
            if val_match:
                val_steps.append(current_step)
                val_mean.append(float(val_match.group(1)))
                val_short.append(float(val_match.group(2)))
                val_med.append(float(val_match.group(3)))
                val_long.append(float(val_match.group(4)))
                continue

            # 3. Egyedi taskok MSE értékei (használjuk a 'short' értékelést reprezentánsként)
            eval_match = eval_pattern.search(line)
            if eval_match:
                scale = eval_match.group(1)
                if scale == "short":  # A tisztább grafikon kedvéért a short értékelést veszzük alapul
                    metrics_str = eval_match.group(2)
                    try:
                        # Gyors dictionary parser a logban lévő dict stringhez
                        metrics = eval("{" + metrics_str + "}")
                        task_steps.append(current_step)
                        task_B.append(metrics.get("B/mse", np.nan))
                        task_T.append(metrics.get("T/mse", np.nan))
                        task_P.append(metrics.get("P/mse", np.nan))
                        task_Z.append(metrics.get("Z/mse", np.nan))
                    except Exception:
                        pass

    return {
        "train": (train_steps, train_loss),
        "val": (val_steps, val_mean, val_short, val_med, val_long),
        "tasks": (task_steps, task_B, task_T, task_P, task_Z),
    }


def plot_training_metrics(data, output_path):
    train_steps, train_loss = data["train"]
    val_steps, val_mean, val_short, val_med, val_long = data["val"]
    task_steps, task_B, task_T, task_P, task_Z = data["tasks"]

    if not train_steps:
        print("Figyelem: Nem található tanítási lépés az adatok között!")
        return

    # Két paneles elrendezés (Fent a Lossok, Lent a Task MSE-k)
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # --- 1. Panel: Train Loss és Val MSE ---
    axes[0].plot(train_steps, train_loss, label="Train Recon Loss", color="tab:blue", alpha=0.8)
    if val_steps:
        axes[0].plot(
            val_steps,
            val_mean,
            label="Val Mean MSE (BTPZ)",
            color="tab:orange",
            marker="o",
            linestyle="--",
            linewidth=2,
        )
        axes[0].plot(val_steps, val_short, label="Val Short", color="tab:gray", alpha=0.5, linestyle=":")
        axes[0].plot(val_steps, val_med, label="Val Medium", color="tab:olive", alpha=0.5, linestyle=":")
        axes[0].plot(val_steps, val_long, label="Val Long", color="tab:purple", alpha=0.5, linestyle=":")

    axes[0].set_ylabel("Loss / MSE")
    axes[0].set_title("F1 SeqBERT: Training Loss & Validation Performance")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # --- 2. Panel: Egyedi Taskok MSE-jei (B, T, P, Z) ---
    if task_steps:
        axes[1].plot(task_steps, task_B, label="Task B MSE (Ghost Car)", marker=".", alpha=0.7)
        axes[1].plot(task_steps, task_T, label="Task T MSE (Tyre Set)", marker=".", alpha=0.7)
        axes[1].plot(task_steps, task_P, label="Task P MSE (Pit-Return)", marker=".", alpha=0.7)
        axes[1].plot(task_steps, task_Z, label="Task Z MSE (Zone)", marker=".", alpha=0.7)

    axes[1].set_xlabel("Training Steps")
    axes[1].set_ylabel("Task MSE")
    axes[1].set_title("Individual Task Validation MSE Over Time")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved successfully as '{output_path}'.")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize F1 SeqBERT training metrics from log files.")
    parser.add_argument(
        "--log",
        type=str,
        default="runs/pre2/logs.jsonl",
        help="Path to the log file (e.g. runs/pre2/logs.jsonl)",
    )
    args = parser.parse_args()

    base, _ = os.path.splitext(args.log)
    output_png = f"{base}_metrics.png"

    data = parse_log(args.log)
    plot_training_metrics(data, output_png)
