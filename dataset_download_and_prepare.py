import os
import tarfile
import urllib.request
from pathlib import Path
from prepare_dataset import main as prepare_main


DATASET_URL = ("https://assets.supervisely.com/remote/"
               "eyJsaW5rIjogInMzOi8vc3VwZXJ2aXNlbHktZGF0YXNldHMvMjc4OF9WZWhpY2xlIERhdGFzZXQgZm9yIFlPTE8vdmVoaWNsZS1kYXRhc2V0LWZvci15b2xvLURhdGFzZXROaW5qYS50YXIiLCAic2lnIjogInRtZEFZaXVzQXZPQkNySVc1L1dXZjVicVY0aS9iUVNnOWJaZlFQMlJzWU09In0=?response-content-disposition=attachment%3B%20filename%3D%22vehicle-dataset-for-yolo-DatasetNinja.tar%22")


def download_dataset(dst_path: Path) -> Path:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if not dst_path.exists():
        print(f"Baixando dataset para {dst_path} ...")
        urllib.request.urlretrieve(DATASET_URL, str(dst_path))
    else:
        print(f"Dataset já existe em {dst_path}")
    return dst_path


def extract_dataset(tar_path: Path, extract_to: Path) -> Path:
    print(f"Extraindo {tar_path} para {extract_to} ...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(tar_path), "r") as tar:
        tar.extractall(path=str(extract_to))
    return extract_to


def main():
    base_dir = Path("c:/Users/pedro/Downloads/YOLO-detection")
    tar_dst = Path("vehicle-dataset-for-yolo-DatasetNinja.tar")
    tar_file = download_dataset(tar_dst)
    extract_to = base_dir
    extract_dataset(tar_file, extract_to)
    print("Convertendo anotações e gerando train.txt/val.txt ...")
    prepare_main()
    print("Concluído.")


if __name__ == "__main__":
    main()

