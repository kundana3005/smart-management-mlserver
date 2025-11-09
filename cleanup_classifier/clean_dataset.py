import os
from PIL import Image

dataset_dir = "dataset"  # your dataset folder

for folder in ["Cleaned", "Uncleaned"]:
    path = os.path.join(dataset_dir, folder)
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            img = Image.open(file_path)
            img.verify()
        except (IOError, SyntaxError) as e:
            print(f"Removing corrupted file: {file_path}")
            os.remove(file_path)