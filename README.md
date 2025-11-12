# YOLO Vehicle Detection - Transfer Learning

Este projeto demonstra como realizar transfer learning para detecção de veículos usando a estrutura Darknet (YOLOv3-tiny) e um dataset personalizado.

## Dataset

O dataset utilizado consiste em 3000 imagens com 3830 objetos rotulados, pertencentes a 6 classes diferentes:

- `car`
- `motorbike`
- `threewheel`
- `van`
- `bus`
- `truck`

O dataset original pode ser encontrado em: [Vehicle Dataset for YOLO](https://assets.supervisely.com/remote/eyJsaW5rIjogInMzOi8vc3VwZXJ2aXNlbHktZGF0YXNldHMvMjc4OF9WZWhpY2xlIERhdGFzZXQgZm9yIFlPTE8vdmVoaWNsZS1kYXRhc2V0LWZvci15b2xvLURhdGFzZXROaW5qYS50YXIiLCAic2lnIjogInRtZEFZaXVzQXZPQkNySVc1L1dXZjVicVY0aS9iUVNnOWJaZlFQMlJzWU09In0=?response-content-disposition=attachment%3B%20filename%3D%22vehicle-dataset-for-yolo-DatasetNinja.tar%22)

## Estrutura do Projeto

```
.gitignore
README.md
prepare_dataset.py
vehicle-dataset-for-yolo-DatasetNinja.tar
darknet/
├── backup/ (diretório para pesos treinados)
├── cfg/
│   ├── obj.data
│   ├── obj.names
│   └── yolov3-tiny-obj.cfg
├── ... (outros arquivos Darknet)
train/
├── ann/ (anotações JSON originais)
└── img/ (imagens de treinamento e anotações YOLO geradas)
val.txt
valid/
├── ann/ (anotações JSON originais)
└── img/ (imagens de validação e anotações YOLO geradas)
```

## Configuração e Preparação

### 1. Clonar o Darknet

Certifique-se de ter o repositório Darknet em seu diretório de trabalho. Se você já tem, pule esta etapa.

```bash
git clone https://github.com/PedroM2626/YOLO-detection.git
```

### 2. Extrair o Dataset

O dataset `vehicle-dataset-for-yolo-DatasetNinja.tar` deve ser extraído no diretório raiz do projeto (`YOLO-detection`).

```bash
tar -xf vehicle-dataset-for-yolo-DatasetNinja.tar
```

### 3. Instalar Dependências Python

O script de preparação do dataset requer `scikit-learn`.

```bash
pip install scikit-learn
```

### 4. Preparar o Dataset para o Darknet

Execute o script `prepare_dataset.py` para converter as anotações JSON para o formato YOLO e gerar os arquivos `train.txt` e `val.txt`.

```bash
python prepare_dataset.py
```

Este script irá:
- Criar arquivos `.txt` com anotações no formato YOLO para cada imagem nas pastas `train/img` e `valid/img`.
- Gerar `train.txt` e `val.txt` na raiz do projeto, listando os caminhos absolutos para as imagens de treinamento e validação, respectivamente.

### 5. Configurar o Darknet

Os seguintes arquivos de configuração foram criados/modificados no diretório `darknet/cfg`:

-   **`obj.names`**: Contém os nomes das 6 classes (car, motorbike, threewheel, van, bus, truck).

    ```
    car
    motorbike
    threewheel
    van
    bus
    truck
    ```

-   **`obj.data`**: Configurações para o Darknet, apontando para os arquivos de treino, validação, nomes das classes e diretório de backup.

    ```
    classes=6
    train = c:/Users/pedro/Downloads/YOLO-detection/train.txt
    valid = c:/Users/pedro/Downloads/YOLO-detection/val.txt
    names = c:/Users/pedro/Downloads/YOLO-detection/darknet/cfg/obj.names
    backup = c:/Users/pedro/Downloads/YOLO-detection/darknet/backup
    ```

-   **`yolov3-tiny-obj.cfg`**: Uma cópia modificada do `yolov3-tiny.cfg` com as seguintes alterações:
    -   `classes=6` nas seções `[yolo]`.
    -   `filters=33` nas camadas `[convolutional]` que precedem as seções `[yolo]` (calculado como `(classes + 5) * 3`).

### 6. Compilar o Darknet

Navegue até o diretório `darknet` e compile-o. Certifique-se de ter o CUDA e o OpenCV instalados se for usar GPU.

```bash
cd darknet
# Para Linux/WSL:
make
# Para Windows, pode ser necessário usar o Visual Studio ou uma versão pré-compilada.
```

## Treinamento

### 1. Baixar Pesos Pré-treinados (Opcional, mas Recomendado)

Para iniciar o treinamento com transfer learning, baixe os pesos pré-treinados do `darknet53.conv.74` e coloque-os no diretório `darknet`.

[darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74)

### 2. Iniciar o Treinamento

No diretório `darknet`, execute o seguinte comando:

```bash
./darknet detector train cfg/obj.data cfg/yolov3-tiny-obj.cfg darknet53.conv.74
```

-   Substitua `./darknet` pelo caminho correto para o executável do Darknet se estiver em Windows ou se o executável não estiver no PATH.
-   Se você não usar pesos pré-treinados, remova `darknet53.conv.74` do comando para treinar do zero.


## Tratamento de Erros e Testes

Este projeto foca na configuração inicial para transfer learning. Para um ambiente de produção, é crucial implementar:

-   **Tratamento de Erros**: Adicionar blocos `try-except` em scripts Python para lidar com erros de arquivo, parsing JSON, etc.
-   **Testes Unitários**: Testar funções individuais, como `convert_bbox_to_yolo` no `prepare_dataset.py`.
-   **Testes de Integração**: Verificar se o pipeline completo de preparação do dataset funciona corretamente, desde a leitura do JSON até a geração dos arquivos `.txt` e `train.txt`/`val.txt`.
-   **Testes de Aceitação**: Validar se o modelo treinado é capaz de detectar objetos nas classes definidas com uma precisão aceitável em um conjunto de dados de teste independente.

## Próximos Passos

-   Monitorar o treinamento e ajustar hiperparâmetros se necessário.
-   Avaliar o desempenho do modelo treinado usando métricas como mAP.
-   Realizar inferência com o modelo treinado em novas imagens/vídeos.