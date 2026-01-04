import os
from pathlib import Path
import yaml
from ultralytics import YOLO

CLASSES = ['car', 'motorbike', 'threewheel', 'van', 'bus', 'truck']


def ensure_data_yaml(base_dir: Path) -> Path:
    data_yaml = base_dir / "data.yaml"
    content = {
        "path": str(base_dir),
        "train": str(base_dir / "train" / "img"),
        "val": str(base_dir / "valid" / "img"),
        "names": CLASSES,
        "nc": len(CLASSES),
    }
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=False, allow_unicode=True)
    return data_yaml


def main():
    base_dir = Path("c:/Users/pedro/Downloads/YOLO-detection")
    data_yaml = ensure_data_yaml(base_dir)
    model = YOLO("yolov8n.pt")
    model.train(data=str(data_yaml), epochs=50, imgsz=640, batch=16)
    model.export(format="onnx")


if __name__ == "__main__":
    main()

