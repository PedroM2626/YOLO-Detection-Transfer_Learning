import os
import pytest
import numpy as np
from yolo_inference import build_detector_from_env


def test_acceptance_basic_pipeline():
    cfg = os.getenv("YOLO_CFG_PATH", "")
    weights = os.getenv("YOLO_WEIGHTS_PATH", "")
    names = os.getenv("YOLO_NAMES_PATH", "")
    if not all(os.path.isfile(p) for p in [cfg, weights, names]):
        pytest.skip("Arquivos de modelo ausentes; aceitação pulada")
    detector = build_detector_from_env()
    img = np.full((480, 640, 3), 255, dtype=np.uint8)
    dets = detector.detect(img)
    assert isinstance(dets, list)
    assert all("class_name" in d for d in dets)
