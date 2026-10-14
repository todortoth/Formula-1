import re

import matplotlib.pyplot as plt


def parse_log(log_path="train.log"):
    train_steps, train_loss = [], []
    train_A, train_B, train_C, train_D = [], [], [], []
    gnorms, lrs = [], []

    val_steps, val_all, val_select = [], [], []
    val_A, val_B, val_C, val_D = [], [], [], []

    # Regular expressions matching train and validation log lines
    train_pattern = re.compile(
        r"step\s+(\d+)\s+\|\s+loss=([\d\.]+)\s+A=([\d\.]+)\s+B=([\d\.]+)\s+C=([\d\.]+)\s+D=([\d\.]+)\s+\|\s+gnorm\s+([\d\.]+)\s+\|\s+lr\s+([\d\.e\-\+]+)"
    )
    val_pattern = re.compile(
        r"\[val\s+@\s+(\d+)\]\s+select\(ABD\)\s+([\d\.]+)\s+\|\s+all\s+([\d\.]+)\s+\|\s+A=([\d\.]+)\s+B=([\d\.]+)\s+C=([\d\.]+)\s+D=([\d\.]+)"
    )

    with open(log_path) as f:
        for line in f:
            train_match = train_pattern.search(line)
            if train_match:
                train_steps.append(int(train_match.group(1)))
                train_loss.append(float(train_match.group(2)))
                train_A.append(float(train_match.group(3)))
                train_B.append(float(train_match.group(4)))
                train_C.append(float(train_match.group(5)))
                train_D.append(float(train_match.group(6)))
                gnorms.append(float(train_match.group(7)))
                lrs.append(float(train_match.group(8)))
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
        "train": (train_steps, train_loss, train_A, train_B, train_C, train_D, gnorms, lrs),
        "val": (val_steps, val_select, val_all, val_A, val_B, val_C, val_D),
    }


def plot_training_metrics(data):
    train_steps, train_loss, train_A, train_B, train_C, train_D, gnorms, lrs = data["train"]
    val_steps, val_select, val_all, val_A, val_B, val_C, val_D = data["val"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1. Total Loss & Validation Metrics
    axes[0].plot(train_steps, train_loss, label="Train Total Loss", color="tab:blue", alpha=0.8)
    if val_steps:
        axes[0].plot(val_steps, val_all, label="Val All", color="tab:orange", marker="o", linestyle="--")
        axes[0].plot(val_steps, val_select, label="Val Select (ABD)", color="tab:green", marker="x", linestyle=":")
    axes[0].set_ylabel("Loss / Metric")
    axes[0].set_title("BERT Pre-training: Loss and Validation Performance")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # 2. Individual Task Losses (A, B, C, D)
    axes[1].plot(train_steps, train_A, label="Task A", alpha=0.7)
    axes[1].plot(train_steps, train_B, label="Task B", alpha=0.7)
    axes[1].plot(train_steps, train_C, label="Task C", alpha=0.7)
    axes[1].plot(train_steps, train_D, label="Task D", alpha=0.7)
    axes[1].set_ylabel("Task Loss")
    axes[1].set_title("Individual Task Losses (Train)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # 3. Learning Rate and Gradient Norm
    ax3_lr = axes[2]
    ax3_gnorm = axes[2].twinx()

    l1 = ax3_lr.plot(train_steps, lrs, label="Learning Rate", color="tab:purple")
    l2 = ax3_gnorm.plot(train_steps, gnorms, label="Gradient Norm (gnorm)", color="tab:red", alpha=0.4)

    axes[2].set_xlabel("Training Steps")
    ax3_lr.set_ylabel("Learning Rate", color="tab:purple")
    ax3_gnorm.set_ylabel("Gradient Norm", color="tab:red")

    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    axes[2].legend(lines, labels, loc="upper right")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./runs/bert_pretraining_metrics.png", dpi=300)
    print("Visualization saved successfully as 'bert_pretraining_metrics.png'.")
    plt.show()


if __name__ == "__main__":
    data = parse_log("train.log")
    plot_training_metrics(data)
