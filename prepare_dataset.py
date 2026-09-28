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

import shutil
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


def main():
    print("--- 1. SETTING UP DIRECTORIES ---")
    base_data_dir = "data_split"
    visual_dir = "output_visualized"
    
    # Create required folder structure
    for split in ['train', 'val']:
        os.makedirs(f"{base_data_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{base_data_dir}/labels/{split}", exist_ok=True)
    os.makedirs(visual_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 2. CREATE DATASET CONFIGURATION (dataset.yaml)
    # ---------------------------------------------------------
    yaml_content = f"""path: ./{base_data_dir}
train: images/train
val: images/val

names:
  0: car
  1: bus
  2: truck
  3: motorcycle
"""
    with open("dataset.yaml", "w") as f:
        f.write(yaml_content)
    print("Created dataset.yaml successfully.")

    # ---------------------------------------------------------
    # 3. AUTO-LABELING & BOUNDING BOX VISUALIZATION
    # ---------------------------------------------------------
    print("\n--- 2. DETECTING OBJECTS & CREATING BOUNDING BOXES ---")
    labeler = YOLO('yolov8n.pt')

    # BGR Colors for bounding box display
    colors = {
        'car': (0, 255, 0),        # Green
        'bus': (0, 0, 255),        # Red
        'truck': (255, 165, 0),    # Blue
        'motorcycle': (0, 255, 255) # Yellow
    }

    coco_to_custom = {2: 0, 5: 1, 7: 2, 3: 3}
    class_names = {0: 'car', 1: 'bus', 2: 'truck', 3: 'motorcycle'}

    source_images = [f for f in os.listdir('images') if f.endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists('images') else []

    if not source_images:
        print("\nError: 'images/' directory not found or empty.")
        print("Please create an 'images/' folder and place your traffic photos inside it.")
        return

    print(f"Found {len(source_images)} image(s) in 'images/'. Processing...")

    for idx, img_file in enumerate(source_images):
        img_path = os.path.join('images', img_file)
        img = cv2.imread(img_path)

        # Split 1 image into val, rest into train
        split = 'val' if idx == 0 else 'train'

        # Copy original image into split directory
        shutil.copy(img_path, f"{base_data_dir}/images/{split}/{img_file}")

        # Run prediction
        results = labeler.predict(source=img_path, conf=0.25, iou=0.45, verbose=False)[0]

        txt_name = os.path.splitext(img_file)[0] + '.txt'
        label_path = f"{base_data_dir}/labels/{split}/{txt_name}"

        with open(label_path, 'w') as f:
            for box in results.boxes:
                cls_id = int(box.cls[0].item())

                if cls_id in coco_to_custom:
                    custom_cls = coco_to_custom[cls_id]
                    label_name = class_names[custom_cls]

                    # Save normalized YOLO annotations (<cls> <x_center> <y_center> <w> <h>)
                    x_center, y_center, bbox_w, bbox_h = box.xywhn[0].tolist()
                    f.write(f"{custom_cls} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")

                    # Draw Bounding Box Rectangle on image
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0].item())
                    box_color = colors.get(label_name, (0, 255, 0))

                    # Bounding box frame
                    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)

                    # Bounding box header tag
                    text = f"{label_name} {conf:.2f}"
                    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w, y1), box_color, -1)
                    cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Save visualized image with boxes
        out_path = os.path.join(visual_dir, f"detected_{img_file}")
        cv2.imwrite(out_path, img)
        print(f"  Processed [{img_file}] -> Labels saved to {label_path}")

    # ---------------------------------------------------------
    # 4. MODEL FINE-TUNING
    # ---------------------------------------------------------
    print("\n--- 3. FINE-TUNING YOLO MODEL ---")
    model = YOLO('yolov8n.pt')

    model.train(
        data='dataset.yaml',
        epochs=15,             # Short epoch count for small sample sets
        imgsz=640,
        batch=2,
        device='cpu',          # Set to 0 if GPU available
        freeze=10,             # Freeze early backbone layers
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        project='runs/train',
        name='traffic_few_shot',
        exist_ok=True
    )

    print("\nTraining complete! Results saved in 'runs/train/traffic_few_shot/'.")

    # ---------------------------------------------------------
    # 5. DISPLAY DETECTED OUTPUT IMAGES
    # ---------------------------------------------------------
    print("\n--- 4. DISPLAYING VISUALIZED OUTPUTS ---")
    output_images = [f for f in os.listdir(visual_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if output_images:
        fig, axes = plt.subplots(1, len(output_images), figsize=(6 * len(output_images), 6))
        if len(output_images) == 1:
            axes = [axes]

        for ax, img_name in zip(axes, output_images):
            img_bgr = cv2.imread(os.path.join(visual_dir, img_name))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(img_name, fontsize=10)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()