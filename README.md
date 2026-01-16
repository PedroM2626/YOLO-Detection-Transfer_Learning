# YOLO Vehicle Detection - Transfer Learning

Este projeto demonstra a detecção de veículos utilizando YOLOv3-tiny com OpenCV DNN. Ele inclui suporte para inferência em imagens e webcam em tempo real, além de uma interface interativa moderna com Streamlit.

## 🚀 Funcionalidades

- **Demo Online**: A interface Gradio está hospedada e pronta para uso em [Hugging Face Spaces](https://huggingface.co/spaces/PedroM2626/YOLO-Detection-Transfer_Learning).
- **Inferência Flexível**: Suporta carregamento de modelos customizados via `.env` ou download automático do YOLOv3-tiny (COCO) como fallback.
- **Interface Streamlit**: Upload de imagens e detecção via webcam em uma interface web amigável.
- **Detecção em Tempo Real**: Script otimizado para webcam com overlays informativos.
- **Notebook Jupyter**: Ambiente para testes rápidos e visualização.
- **Preparação de Dataset**: Conversão de anotações do formato JSON para o padrão YOLO.

## 🛠️ Instalação

1.  **Clone o repositório**:
    ```bash
    git clone <url-do-repositorio>
    cd YOLO-Detection-Transfer_Learning
    ```

2.  **Instale as dependências**:
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Configuração

Crie um arquivo `.env` na raiz do projeto (ou use o `.env.example`) para configurar os caminhos do seu modelo treinado:

```env
YOLO_CFG_PATH=models/yolov3-tiny.cfg
YOLO_WEIGHTS_PATH=models/yolov3-tiny.weights
YOLO_NAMES_PATH=models/coco.names
YOLO_CONF_THRESHOLD=0.5
YOLO_NMS_THRESHOLD=0.4
YOLO_USE_GPU=false
```

*Nota: Se os arquivos não forem encontrados nos caminhos acima, o sistema baixará automaticamente o modelo YOLOv3-tiny padrão para a pasta `models/`.*

## 🖥️ Como Usar

### 1. Interface Streamlit (Recomendado)
A interface web permite testar imagens e webcam facilmente:
```bash
streamlit run app_streamlit.py
```

### 2. Detecção via Webcam (CLI)
Para uma execução direta via terminal:
```bash
python yolo_realtime.py
```

### 3. Inferência em Imagem (CLI)
```bash
python yolo_inference.py --image caminho/para/imagem.jpg
```

### 4. Preparação do Dataset
Se você tiver o dataset original em JSON:
```bash
python prepare_dataset.py
```

## 📁 Estrutura do Projeto

- `app_streamlit.py`: Interface web interativa.
- `yolo_inference.py`: Core da lógica de detecção e gerenciamento de modelos.
- `yolo_realtime.py`: Script para execução em tempo real via terminal.
- `prepare_dataset.py`: Utilitário para conversão de anotações.
- `notebooks/yolo_notebook.ipynb`: Demonstração em ambiente Jupyter.
- `models/`: Pasta onde os pesos e configurações são armazenados/baixados.

## 📝 Notas
- O projeto utiliza **caminhos relativos** para garantir portabilidade.
- O detector prioriza classes como `car`, `truck`, `bus`, `motorbike` e `van`.
- Pressione **'q'** para sair das janelas de visualização OpenCV.
