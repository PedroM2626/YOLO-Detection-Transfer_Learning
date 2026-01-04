import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def _load_classes(names_path: str) -> List[str]:
    if not os.path.isfile(names_path):
        raise FileNotFoundError(f"Arquivo de classes não encontrado: {names_path}")
    with open(names_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    if not classes:
        raise ValueError("Lista de classes vazia")
    return classes


def _get_output_layer_names(net: cv2.dnn_Net) -> List[str]:
    layer_names = net.getLayerNames()
    out_layers = net.getUnconnectedOutLayers()
    return [layer_names[i - 1] for i in out_layers.flatten()]


class YoloDetector:
    def __init__(
        self,
        cfg_path: str,
        weights_path: str,
        names_path: str,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        use_gpu: bool = False,
    ):
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"CFG não encontrado: {cfg_path}")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Pesos não encontrados: {weights_path}")
        self.classes = _load_classes(names_path)
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        if use_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.output_layer_names = _get_output_layer_names(self.net)

    def detect(
        self,
        image_bgr: np.ndarray,
        input_size: Tuple[int, int] = (416, 416),
    ) -> List[Dict]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Imagem inválida para detecção")
        h, w = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(image_bgr, 1 / 255.0, input_size, swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layer_names)

        boxes: List[List[int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence >= self.conf_threshold:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    width = int(detection[2] * w)
                    height = int(detection[3] * h)
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    boxes.append([x, y, width, height])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        detections: List[Dict] = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w_box, h_box = boxes[i]
                detections.append(
                    {
                        "class_id": class_ids[i],
                        "class_name": self.classes[class_ids[i]] if 0 <= class_ids[i] < len(self.classes) else str(class_ids[i]),
                        "confidence": confidences[i],
                        "box": (max(0, x), max(0, y), max(0, w_box), max(0, h_box)),
                    }
                )
        return detections

    def draw(self, image_bgr: np.ndarray, detections: List[Dict]) -> np.ndarray:
        out = image_bgr.copy()
        for det in detections:
            x, y, w, h = det["box"]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x, y - th - 6), (x + tw + 4, y), color, -1)
            cv2.putText(out, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return out


def build_detector_from_env(
    conf_threshold: Optional[float] = None,
    nms_threshold: Optional[float] = None,
    use_gpu: Optional[bool] = None,
) -> YoloDetector:
    if load_dotenv is not None:
        load_dotenv()
    cfg_path = os.getenv("YOLO_CFG_PATH", "").strip()
    weights_path = os.getenv("YOLO_WEIGHTS_PATH", "").strip()
    names_path = os.getenv("YOLO_NAMES_PATH", "").strip()
    if not cfg_path or not weights_path or not names_path:
        raise EnvironmentError("Variáveis de ambiente YOLO_CFG_PATH, YOLO_WEIGHTS_PATH e YOLO_NAMES_PATH são obrigatórias")
    ct = float(os.getenv("YOLO_CONF_THRESHOLD", conf_threshold if conf_threshold is not None else 0.5))
    nt = float(os.getenv("YOLO_NMS_THRESHOLD", nms_threshold if nms_threshold is not None else 0.4))
    gpu_flag = os.getenv("YOLO_USE_GPU", "false").lower() in {"1", "true", "yes"} if use_gpu is None else use_gpu
    return YoloDetector(cfg_path=cfg_path, weights_path=weights_path, names_path=names_path, conf_threshold=ct, nms_threshold=nt, use_gpu=gpu_flag)

