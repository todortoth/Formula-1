#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from f1strategy import ParquetSequence, RaceAnimator


def verify_and_animate_dataset(dataset_dir: str | Path, output_gif: str | Path, target_fps: float):
    """
    Loads an existing disk Parquet dataset, verifies its continuity matrix,
    and outputs a viewable, high-speed diagnostic GIF.
    """
    dataset_path = Path(dataset_dir)

    # 1. Safety Check: Verify the manifest exists
    if not (dataset_path / "manifest.json").exists():
        raise FileNotFoundError(
            f"Could not find manifest.json in '{dataset_path}'. "
            "If the scraping script crashed or was killed midway, this file will be missing."
        )

    print(f"--- Opening Dataset: {dataset_path.name} ---")
    ds = ParquetSequence.open(dataset_path)

    # 2. Run the pipeline's built-in continuity & health checks
    print("Running data integrity checks...")
    report = ds.continuity_report()
    print(report.summary())

    if not report.ok:
        print("[WARNING]: Continuity anomalies detected (e.g., driver teleports or timeline steps).")
    else:
        print("[SUCCESS]: Data physics and continuity look solid for model training.")

    # 3. Calculate frames and playback speed for a watchable GIF
    total_frames = ds.n_frames

    # Target roughly ~350 frames total for a perfectly watchable diagnostic GIF
    max_desired_gif_frames = 350
    every = max(1, -(-total_frames // max_desired_gif_frames))

    # Real-time speed calculation: (every * step_size_seconds * fps)
    # The source dataset is 10 Hz, meaning step_size_seconds is exactly 0.1s
    realtime_multiplier = every * 0.1 * target_fps

    print(f"\nDataset contains {total_frames} total frames (sampled at 10 Hz).")
    print(f"Decimating sequence: Loading every {every}th frame ({total_frames // every} frames total).")
    print(f"GIF Configuration: {target_fps} FPS (Playing at {realtime_multiplier:.1f}x real-time speed).")

    # 4. Read only the thinned-out sequence frames directly from the Parquet files
    seq_gif = ds.read_sequence(start_frame=0, stop_frame=None, every=every)

    # 5. Render the visual animation layout
    print(f"Rendering diagnostic GIF to '{output_gif}'...")

    # Resolve out folder if it doesn't exist yet
    os.makedirs(Path(output_gif).parent, exist_ok=True)

    title_label = getattr(ds, 'location', dataset_path.name)

    animator = RaceAnimator(
        sequence=seq_gif,
        colors=None,  # Automatically falls back to internal team colors if available
        title=f"Verification Check: {title_label}"
    )

    animator.save_gif(output_gif, every=1, fps=target_fps)
    print(f"Verification complete. GIF saved to: {output_gif}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify an existing F1 Parquet dataset and generate a fast diagnostic GIF."
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to the target dataset directory (e.g., parquet_datasets/2025_round_1.ds)"
    )
    parser.add_argument(
        "--out-gif", "-o",
        type=str,
        default=None,
        help="Path where the verification GIF should be saved. (Default: inside a 'graphs' folder next to the dataset)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=12.0,
        help="Target playback frames per second for the GIF output (Default: 12.0)"
    )

    args = parser.parse_args()

    # Handle dynamic fallback naming if no explicit output path is specified
    if args.out_gif is None:
        dataset_name = Path(args.dataset).stem
        args.out_gif = f"graphs/{dataset_name}_verification.gif"

    try:
        verify_and_animate_dataset(
            dataset_dir=args.dataset,
            output_gif=args.out_gif,
            target_fps=args.fps
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
