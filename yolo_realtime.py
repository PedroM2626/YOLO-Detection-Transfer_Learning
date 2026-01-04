import os
import sys
import argparse
import cv2
from yolo_inference import build_detector_from_env, YoloDetector

# Lista de classes do dataset custom utilizado no projeto
CLASSES = ['car', 'motorbike', 'threewheel', 'van', 'bus', 'truck']

def parse_args() -> argparse.Namespace:
    # Parser de argumentos CLI para configurar câmera, resolução, thresholds e caminhos
    parser = argparse.ArgumentParser(description="Detecção em tempo real com YOLO (OpenCV DNN)")
    parser.add_argument("--camera", type=int, default=0, help="Índice da câmera (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Largura do frame")
    parser.add_argument("--height", type=int, default=720, help="Altura do frame")
    parser.add_argument("--input-size", type=str, default="416x416", help="Tamanho de entrada da rede, ex: 416x416")
    parser.add_argument("--conf", type=float, default=None, help="Confiança mínima")
    parser.add_argument("--nms", type=float, default=None, help="NMS threshold")
    parser.add_argument("--gpu", action="store_true", help="Usar CUDA (se disponível)")
    parser.add_argument("--cfg", type=str, default=None, help="Caminho para .cfg (sobrepõe .env)")
    parser.add_argument("--weights", type=str, default=None, help="Caminho para .weights (sobrepõe .env)")
    parser.add_argument("--names", type=str, default=None, help="Caminho para .names (sobrepõe .env)")
    return parser.parse_args()


def make_detector(args: argparse.Namespace) -> YoloDetector:
    # Inicializa o detector: usa caminhos explícitos se fornecidos, senão .env/fallback
    if args.cfg and args.weights and args.names:
        return YoloDetector(
            cfg_path=args.cfg,
            weights_path=args.weights,
            names_path=args.names,
            conf_threshold=args.conf if args.conf is not None else 0.5,
            nms_threshold=args.nms if args.nms is not None else 0.4,
            use_gpu=args.gpu,
        )
    return build_detector_from_env(conf_threshold=args.conf, nms_threshold=args.nms, use_gpu=args.gpu)


def main() -> int:
    # Loop principal de captura da câmera, inferência e exibição com overlay
    args = parse_args()
    try:
        detector = make_detector(args)
    except Exception as e:
        print(f"Erro ao inicializar o detector: {e}")
        return 2

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Não foi possível abrir a câmera")
        return 3
    if args.width > 0 and args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    try:
        w_str, h_str = args.input_size.lower().split("x")
        input_size = (int(w_str), int(h_str))
    except Exception:
        input_size = (416, 416)

    print("Pressione 'q' para sair")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar frame")
            break
        try:
            detections = detector.detect(frame, input_size=input_size)
            frame_out = detector.draw(frame, detections)
        except Exception as e:
            print(f"Erro na detecção: {e}")
            frame_out = frame
        # Desenha instrução de saída na tela
        overlay1 = "Pressione 'q' para sair"
        (tw1, th1), _ = cv2.getTextSize(overlay1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame_out, (8, 8), (8 + tw1 + 6, 8 + th1 + 10), (0, 0, 0), -1)
        cv2.putText(frame_out, overlay1, (12, 8 + th1 + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # Coleta classes detectadas do dataset custom e exibe overlay se houver
        hits = sorted({d['class_name'] for d in detections if d.get('class_name') in CLASSES})
        if hits:
            overlay2 = "Detectadas: " + ", ".join(hits)
            (tw2, th2), _ = cv2.getTextSize(overlay2, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            y0 = 20 + th1 + 16
            cv2.rectangle(frame_out, (8, y0), (8 + tw2 + 6, y0 + th2 + 8), (0, 0, 0), -1)
            cv2.putText(frame_out, overlay2, (12, y0 + th2 + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("YOLO - Detecção em tempo real", frame_out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
