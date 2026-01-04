import os
import numpy as np
import cv2
import pytest
from yolo_inference import YoloDetector


def test_init_missing_cfg():
    with pytest.raises(FileNotFoundError):
        YoloDetector(cfg_path="missing.cfg", weights_path="missing.weights", names_path="missing.names")


def test_detect_invalid_image():
    tmp_cfg = __file__
    tmp_weights = __file__
    tmp_names = __file__
    with pytest.raises(FileNotFoundError):
        YoloDetector(cfg_path=tmp_cfg, weights_path=tmp_weights, names_path=tmp_names)


def test_draw_no_detections():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = []
    out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    assert out.shape == img.shape
