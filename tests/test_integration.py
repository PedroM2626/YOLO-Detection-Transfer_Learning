import os
import pytest
import numpy as np
from yolo_inference import build_detector_from_env


def has_model_files() -> bool:
    cfg = os.getenv("YOLO_CFG_PATH", "")
    weights = os.getenv("YOLO_WEIGHTS_PATH", "")
    names = os.getenv("YOLO_NAMES_PATH", "")
    return all(os.path.isfile(p) for p in [cfg, weights, names])


def test_build_detector_from_env_or_skip():
    if not has_model_files():
        pytest.skip("Arquivos de modelo ausentes; integração pulada")
    detector = build_detector_from_env()
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    dets = detector.detect(img)
    assert isinstance(dets, list)
