# Importações necessárias para Streamlit, OpenCV e processamento de imagem
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from yolo_inference import build_detector_from_env

# Configuração inicial da página do Streamlit (Título e Layout)
st.set_page_config(page_title="YOLO Detection - Streamlit", layout="wide", page_icon="🚗")

def main():
    """
    Função principal que gerencia a interface Streamlit.
    Permite alternar entre detecção em imagens estáticas e vídeo em tempo real via webcam.
    """
    st.title("🚀 YOLO Object Detection")
    st.markdown("---")
    st.markdown("### Interface interativa para detecção de objetos usando YOLOv3-tiny.")

    # Sidebar: Painel lateral para controle de parâmetros e seleção de modo
    st.sidebar.header("🛠️ Configurações do Modelo")
    
    # Sliders para ajuste dinâmico dos limiares de detecção
    conf_threshold = st.sidebar.slider("Confiança Mínima (Threshold)", 0.0, 1.0, 0.5, 0.05, 
                                     help="Nível mínimo de certeza para exibir uma detecção.")
    nms_threshold = st.sidebar.slider("NMS Threshold", 0.0, 1.0, 0.4, 0.05,
                                    help="Limiar para supressão de não-máximos (remove bboxes sobrepostas).")
    
    st.sidebar.markdown("---")
    # Seleção do modo de operação
    mode = st.sidebar.radio("📡 Escolha o Modo de Entrada", ["Imagem", "Câmera (Real-time)"])

    # Inicializa o detector YOLO
    # A função build_detector_from_env gerencia o download automático dos pesos se necessário.
    try:
        detector = build_detector_from_env(conf_threshold=conf_threshold, nms_threshold=nms_threshold)
    except Exception as e:
        st.error(f"❌ Erro ao inicializar detector: {e}")
        return

    # Lista de classes do dataset personalizado para monitoramento especial
    CUSTOM_CLASSES = {"car", "truck", "bus", "motorbike", "bicycle", "van", "threewheel"}

    if mode == "Imagem":
        st.subheader("📁 Upload e Detecção em Imagem")
        uploaded_file = st.file_uploader("Arraste ou selecione uma imagem...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Converte o arquivo carregado (BytesIO) para uma imagem PIL e depois para array numpy
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Streamlit/PIL trabalham em RGB, mas o detector OpenCV espera BGR
            frame_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Realiza a detecção de objetos
            with st.spinner('Processando imagem...'):
                detections = detector.detect(frame_bgr)
            
            # Filtra e exibe classes encontradas que pertencem ao dataset customizado
            hits = sorted({d['class_name'] for d in detections if d['class_name'] in CUSTOM_CLASSES})
            
            # Layout em duas colunas: Imagem original vs Resultado
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Imagem Original", use_column_width=True)
            
            with col2:
                # Desenha os retângulos e labels no frame BGR
                result_bgr = detector.draw(frame_bgr, detections)
                # Converte de volta para RGB para exibição correta no Streamlit
                result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, caption="Detecções Encontradas", use_column_width=True)

            # Exibe alertas baseados nas classes detectadas
            if hits:
                st.success(f"✅ Objetos do dataset detectados: **{', '.join(hits)}**")
            else:
                st.info("ℹ️ Nenhuma classe do dataset específico foi detectada nesta imagem.")

    elif mode == "Câmera (Real-time)":
        st.subheader("🎥 Detecção via Webcam em Tempo Real")
        st.warning("⚠️ Certifique-se de que sua webcam não está sendo usada por outro aplicativo.")
        
        # Checkbox para ligar/desligar o loop da câmera
        run = st.checkbox("Ativar Câmera")
        
        # Placeholders para atualização dinâmica do frame e status sem recarregar a página toda
        frame_placeholder = st.empty()
        status_placeholder = st.empty()

        if run:
            # Inicializa a captura de vídeo (ID 0 costuma ser a webcam padrão)
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Não foi possível acessar a câmera. Verifique as permissões.")
                return

            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Falha ao capturar vídeo.")
                    break

                # Processa o frame atual
                detections = detector.detect(frame)
                
                # Renderiza as detecções no frame
                frame_out = detector.draw(frame, detections)
                
                # Adiciona overlay de instrução no frame (estilo solicitado anteriormente)
                cv2.putText(frame_out, "Desmarque 'Ativar Camera' para sair", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Identifica classes do dataset para exibição de status dinâmico
                hits = sorted({d['class_name'] for d in detections if d['class_name'] in CUSTOM_CLASSES})
                if hits:
                    status_placeholder.success(f"Detectado: **{', '.join(hits)}**")
                else:
                    status_placeholder.empty()

                # Conversão BGR -> RGB para o Streamlit renderizar corretamente
                frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                # Pequeno delay opcional para sincronia (cv2.waitKey não é necessário aqui para exibição, 
                # mas ajuda a liberar CPU)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Libera recursos ao encerrar
            cap.release()
            st.write("🏁 Captura encerrada.")
        else:
            st.write("💤 Câmera em espera.")

if __name__ == "__main__":
    main()

