import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def parse_log(log_path):
    train_steps, train_loss = [], []

    # Validációs adatok tárolói (a felső és alsó panelhez)
    val_steps = []
    val_mean = []

    val_task_steps = []
    val_B, val_T, val_P, val_Z = [], [], [], []

    print(f"JSONL log feldolgozása: {log_path}")

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            step = data.get("step")
            if step is None:
                continue

            # 1. Train adatok (Kék folytonos vonal a felső panelen)
            loss = data.get("loss")
            if loss is not None:
                train_steps.append(step)
                train_loss.append(float(loss))

            # 2. Validációs átlag adatok (Narancssárga szaggatott vonal a felső panelen)
            # MEGJEGYZÉS: Cseréld ki a "val_loss" kulcsot arra, ami a logfájlodban szerepel (pl. val_mean_mse)
            v_mean = data.get("val_loss") or data.get("val_mean_mse")
            if v_mean is not None:
                val_steps.append(step)
                val_mean.append(float(v_mean))

            # 3. Egyedi Task Validációs MSE-k (Alsó panel vonalai)
            # MEGJEGYZÉS: Cseréld ki a kulcsokat a logfájlodban szereplő pontos validációs kulcsokra (pl. val_loss_B vagy val_mse_B)
            b = data.get("val_loss_B") or data.get("loss_B")
            t = data.get("val_loss_T") or data.get("loss_T")
            p = data.get("val_loss_P") or data.get("loss_P")
            z = data.get("val_loss_Z") or data.get("loss_Z")

            # Csak akkor mentjük el validációs pontként, ha ez egy validációs lépés volt (pl. ritkábban fut, mint a train)
            if any(v is not None for v in [b, t, p, z]) and (data.get("val_loss") is not None or "val" in data):
                val_task_steps.append(step)
                val_B.append(float(b) if b is not None else np.nan)
                val_T.append(float(t) if t is not None else np.nan)
                val_P.append(float(p) if p is not None else np.nan)
                val_Z.append(float(z) if z is not None else np.nan)

    return {
        "train": (train_steps, train_loss),
        "val_mean": (val_steps, val_mean),
        "val_tasks": (val_task_steps, val_B, val_T, val_P, val_Z),
    }


def plot_training_metrics(data, output_path):
    train_steps, train_loss = data["train"]
    val_steps, val_mean = data["val_mean"]
    val_task_steps, val_B, val_T, val_P, val_Z = data["val_tasks"]

    if not train_steps:
        print("Figyelem: Nem található tanítási lépés az adatok között!")
        return

    # Két paneles elrendezés (Megnövelt szélesség a mintakép arányaihoz)
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # --- 1. Panel: Train Loss & Validation Performance ---
    # Kék vékony folytonos vonal (eredeti train loss, simítás nélkül, mert a mintán is az van)
    axes[0].plot(
        train_steps,
        train_loss,
        label="Tanítási rekonstrukciós veszteség (BTPZ)",
        color="tab:blue",
        alpha=0.7,
        linestyle="-",
        linewidth=1.2,
    )

    # Narancssárga szaggatott vonal nagy kör pontokkal (Validációs átlag)
    if val_steps:
        axes[0].plot(
            val_steps,
            val_mean,
            label="Validációs átlagos MSE (BTPZ)",
            color="tab:orange",
            linestyle="--",
            linewidth=1.8,
            marker="o",
            markersize=6,
        )

    axes[0].set_ylabel("Veszteség / MSE")
    axes[0].set_title("F1 SeqBERT: Tanítási veszteség és validációs teljesítmény")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.2, linestyle="-")

    # --- 2. Panel: Individual Task Validation MSE Over Time ---
    if val_task_steps:
        # B feladat - Kék vonal pontokkal
        axes[1].plot(
            val_task_steps,
            val_B,
            label="B feladat MSE (Ghost Car)",
            color="tab:blue",
            alpha=0.8,
            linestyle="-",
            linewidth=1.5,
            marker="o",
            markersize=4,
        )
        # T feladat - Narancs vonal pontokkal
        axes[1].plot(
            val_task_steps,
            val_T,
            label="T feladat MSE (Tyre Set)",
            color="tab:orange",
            alpha=0.8,
            linestyle="-",
            linewidth=1.5,
            marker="o",
            markersize=4,
        )
        # P feladat - Zöld vonal pontokkal
        axes[1].plot(
            val_task_steps,
            val_P,
            label="P feladat MSE (Pit-Return)",
            color="tab:green",
            alpha=0.8,
            linestyle="-",
            linewidth=1.5,
            marker="o",
            markersize=4,
        )
        # Z feladat - Piros vonal pontokkal
        axes[1].plot(
            val_task_steps,
            val_Z,
            label="Z feladat MSE (Zone)",
            color="tab:red",
            alpha=0.8,
            linestyle="-",
            linewidth=1.5,
            marker="o",
            markersize=4,
        )

    axes[1].set_xlabel("Tanítási lépések (Steps)")
    axes[1].set_ylabel("Feladat MSE")
    axes[1].set_title("Egyedi feladatok validációs MSE értékei az időben")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.2, linestyle="-")

    # X tengely skálázás beállítása fixen 0-tól indulva, 5000-es felirat lépésközzel
    all_steps = train_steps + val_steps
    max_step = max(all_steps) if all_steps else 30000

    axes[1].set_xlim(left=0, right=max_step + 500)
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(5000))

    # Háttérrács finomítása a tiszta fehér/világosszürke hatáshoz
    for ax in axes:
        ax.set_facecolor("#ffffff")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"A diagram sikeresen elmentve '{output_path}' néven!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize F1 SeqBERT training metrics from JSONL log files.")
    parser.add_argument(
        "--log",
        type=str,
        default="runs/mixed_gnn/log.jsonl",
        help="Path to the log file (e.g. runs/mixed_gnn/log.jsonl)",
    )
    args = parser.parse_args()

    base, _ = os.path.splitext(args.log)
    output_png = f"{base}_metrics.png"

    data = parse_log(args.log)
    plot_training_metrics(data, output_png)
