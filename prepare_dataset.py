import os
import json
from sklearn.model_selection import train_test_split

# Define as classes na ordem correta
CLASSES = ['car', 'motorbike', 'threewheel', 'van', 'bus', 'truck']

def convert_bbox_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def process_annotations(image_dir, ann_dir, image_paths_list):
    for ann_file in os.listdir(ann_dir):
        if ann_file.endswith('.json'):
            json_path = os.path.join(ann_dir, ann_file)
            with open(json_path, 'r') as f:
                data = json.load(f)

            image_filename_base = os.path.splitext(ann_file)[0] # Get '1.jpg' from '1.jpg.json'
            image_path = os.path.join(image_dir, image_filename_base)
            
            # Verifica se a imagem existe antes de processar
            if not os.path.exists(image_path):
                print(f"Aviso: Imagem {image_path} não encontrada. Pulando anotação {ann_file}.")
                continue

            # Adiciona o caminho da imagem à lista
            image_paths_list.append(os.path.abspath(image_path))

            img_w = data['size']['width']
            img_h = data['size']['height']

            yolo_annotations = []
            for obj in data['objects']:
                class_name = obj['classTitle']
                if class_name in CLASSES:
                    class_id = CLASSES.index(class_name)
                    
                    # Coordenadas da bounding box
                    x1 = obj['points']['exterior'][0][0]
                    y1 = obj['points']['exterior'][0][1]
                    x2 = obj['points']['exterior'][1][0]
                    y2 = obj['points']['exterior'][1][1]

                    b = (x1, x2, y1, y2) # (x_min, x_max, y_min, y_max)
                    bb = convert_bbox_to_yolo((img_w, img_h), b)
                    yolo_annotations.append(f"{class_id} {bb[0]} {bb[1]} {bb[2]} {bb[3]}")
                else:
                    print(f"Aviso: Classe '{class_name}' não encontrada em CLASSES. Pulando anotação em {ann_file}.")

            # Salva as anotações YOLO no mesmo diretório da imagem
            output_annotation_path = os.path.join(image_dir, os.path.splitext(image_filename_base)[0] + '.txt')
            with open(output_annotation_path, 'w') as f:
                for line in yolo_annotations:
                    f.write(line + '\n')

def main():
    base_dir = 'c:/Users/pedro/Downloads/YOLO-detection'
    train_img_dir = os.path.join(base_dir, 'train', 'img')
    train_ann_dir = os.path.join(base_dir, 'train', 'ann')
    valid_img_dir = os.path.join(base_dir, 'valid', 'img')
    valid_ann_dir = os.path.join(base_dir, 'valid', 'ann')

    all_image_paths = []

    print("Processando anotações de treinamento...")
    process_annotations(train_img_dir, train_ann_dir, all_image_paths)
    print("Processando anotações de validação...")
    process_annotations(valid_img_dir, valid_ann_dir, all_image_paths)

    # Dividir em treino e validação (80/20)
    train_paths, val_paths = train_test_split(all_image_paths, test_size=0.2, random_state=42)

    with open(os.path.join(base_dir, 'train.txt'), 'w') as f:
        for path in train_paths:
            f.write(path + '\n')

    with open(os.path.join(base_dir, 'val.txt'), 'w') as f:
        for path in val_paths:
            f.write(path + '\n')

    print("Dataset preparado com sucesso!")
    print(f"Total de imagens processadas: {len(all_image_paths)}")
    print(f"Imagens de treinamento: {len(train_paths)}")
    print(f"Imagens de validação: {len(val_paths)}")

if __name__ == '__main__':
    main()