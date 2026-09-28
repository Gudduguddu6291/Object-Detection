import os
import sys
from pathlib import Path


def ensure_project_venv():
    project_root = Path(__file__).resolve().parent
    venv_python = project_root / ".venv-1" / "Scripts" / "python.exe"
    current_python = Path(sys.executable).resolve()
    project_venv_root = project_root / ".venv-1"

    is_in_project_venv = str(current_python).startswith(str(project_venv_root.resolve()))
    if venv_python.exists() and not is_in_project_venv:
        print(f"Using project virtual environment: {venv_python}")
        os.execv(str(venv_python), [str(venv_python), __file__, *sys.argv[1:]])


ensure_project_venv()

from ultralytics import YOLO


def train_small_dataset():
    # Load lightweight model pre-trained on COCO
    model = YOLO('yolov8n.pt')

    # Fine-tune with transfer learning and heavy augmentation
    results = model.train(
        data='dataset.yaml',
        epochs=30,             # Keep epochs low to prevent overfitting on 3 images
        imgsz=640,
        batch=2,               # Small batch size for small dataset
        device='cpu',          # Set to 0 if using GPU (e.g., device=0)
        
        # Transfer learning settings
        freeze=10,             # Freeze early backbone layers to preserve general feature extraction
        
        # Augmentation hyperparameters to avoid overfitting
        hsv_h=0.015,           # Color jitter
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,          # Small rotations
        translate=0.1,         # Translation
        scale=0.5,             # Scaling
        fliplr=0.5,            # Horizontal flip
        mosaic=1.0,            # Combine parts of multiple images into one
        
        project='runs/train',
        name='traffic_few_shot'
    )

    # Validate
    metrics = model.val()
    print(f"Fine-tuned mAP@50: {metrics.box.map50:.4f}")

    # Test inference on train image
    model.predict(source='images/img3.jpg', save=True, conf=0.3)

if __name__ == '__main__':
    train_small_dataset()