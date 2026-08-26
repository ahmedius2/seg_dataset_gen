"""
Train an Ultralytics YOLO11 nano instance-segmentation model.

Run from the activated project environment:
    source blenderproc_py_venv/bin/activate
    python train_yolo11_seg.py

The dataset must first be generated with convert_to_yolo.py.
"""

from pathlib import Path


# Training configuration
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_YAML = PROJECT_ROOT / "yolo_dataset" / "dataset.yaml"
MODEL_WEIGHTS = "yolo11n-seg.pt"
RUNS_DIR = PROJECT_ROOT / "runs" / "segment"
RUN_NAME = "yolo11n_seg_drone"

EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 16
WORKERS = 4
DEVICE = "0"  # Use "cpu" for CPU training or "0" for the first GPU.

# Reproducibility and checkpointing
SEED = 42
PATIENCE = 30
SAVE_PERIOD = -1  # Save only the best and last checkpoints.


def main() -> None:
    """Validate inputs and train YOLO11-seg nano."""
    if not DATASET_YAML.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATASET_YAML}\n"
            "Run convert_to_yolo.py before starting training."
        )

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is not installed in the active environment. "
            "Install it with: python -m pip install ultralytics"
        ) from error

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_WEIGHTS)
    model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        seed=SEED,
        patience=PATIENCE,
        save_period=SAVE_PERIOD,
        pretrained=True,
        task="segment",
    )


if __name__ == "__main__":
    main()
