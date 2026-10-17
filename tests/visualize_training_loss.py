import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path):
    train_steps = []
    train_loss = []
    train_B, train_T, train_P, train_Z = [], [], [], []

    val_steps, val_all, val_select = [], [], []
    val_A, val_B, val_C, val_D = [], [], [], []

    if log_path.endswith(".jsonl"):
        print(f"JSONL log feldolgozása: {log_path}")
        steps_dict = {}

        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                step = data.get("step", data.get("train_step"))
                if step is not None:
                    step = int(step)
                    if step not in steps_dict:
                        steps_dict[step] = {}
                    steps_dict[step].update(data)

        sorted_steps = sorted(steps_dict.keys())
        for s in sorted_steps:
            d = steps_dict[s]
            if "loss" in d or "train_loss" in d:
                train_steps.append(s)
                train_loss.append(float(d.get("loss", d.get("train_loss", np.nan))))

                train_B.append(float(d.get("loss_B", np.nan)))
                train_T.append(float(d.get("loss_T", np.nan)))
                train_P.append(float(d.get("loss_P", np.nan)))
                train_Z.append(float(d.get("loss_Z", np.nan)))

    else:
        print(f"Szöveges log (regex) feldolgozása: {log_path}")
        train_pattern = re.compile(
            r"step\s+(\d+)\s+\|\s+loss=([\d\.]+)\s+A=([\d\.]+)\s+B=([\d\.]+)\s+C=([\d\.]+)\s+D=([\d\.]+)\s+\|\s+gnorm\s+([\d\.]+)\s+\|\s+lr\s+([\d\.e\-\+]+)"
        )
        val_pattern = re.compile(
            r"\[val\s+@\s+(\d+)\]\s+select\(ABD\)\s+([\d\.]+)\s+\|\s+all\s+([\d\.]+)\s+\|\s+A=([\d\.]+)\s+B=([\d\.]+)\s+C=([\d\.]+)\s+D=([\d\.]+)"
        )

        with open(log_path, encoding="utf-8") as f:
            for line in f:
                train_match = train_pattern.search(line)
                if train_match:
                    train_steps.append(int(train_match.group(1)))
                    train_loss.append(float(train_match.group(2)))
                    train_B.append(float(train_match.group(4)))
                    train_T.append(float(train_match.group(3)))
                    train_P.append(float(train_match.group(5)))
                    train_Z.append(float(train_match.group(6)))
                    continue

                val_match = val_pattern.search(line)
                if val_match:
                    val_steps.append(int(val_match.group(1)))
                    val_select.append(float(val_match.group(2)))
                    val_all.append(float(val_match.group(3)))
                    val_A.append(float(val_match.group(4)))
                    val_B.append(float(val_match.group(5)))
                    val_C.append(float(val_match.group(6)))
                    val_D.append(float(val_match.group(7)))

    return {
        "train": (train_steps, train_loss, train_B, train_T, train_P, train_Z),
        "val": (val_steps, val_select, val_all, val_A, val_B, val_C, val_D),
    }


def plot_training_metrics(data, output_path):
    train_steps, train_loss, train_B, train_T, train_P, train_Z = data["train"]
    val_steps, val_select, val_all, val_A, val_B, val_C, val_D = data["val"]

    if not train_steps:
        print("Figyelem: Nem található tanítási lépés az adatok között!")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(train_steps, train_loss, label="Train Total Loss", color="tab:blue", alpha=0.8)
    if val_steps:
        axes[0].plot(val_steps, val_all, label="Val All", color="tab:orange", marker="o", linestyle="--")
        axes[0].plot(val_steps, val_select, label="Val Select (ABD)", color="tab:green", marker="x", linestyle=":")
    axes[0].set_ylabel("Loss / Metric")
    axes[0].set_title("BERT Pre-training: Loss and Validation Performance")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_steps, train_B, label="Task B (Ghost Car)", alpha=0.6)
    axes[1].plot(train_steps, train_T, label="Task T (Tyre Set)", alpha=0.6)
    axes[1].plot(train_steps, train_P, label="Task P (Pit-Return)", alpha=0.6)
    axes[1].plot(train_steps, train_Z, label="Task Z (Zone)", alpha=0.6)
    axes[1].set_xlabel("Training Steps")
    axes[1].set_ylabel("Task Loss")
    axes[1].set_title("Individual Task Losses (Train)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved successfully as '{output_path}'.")
    plt.show()


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
