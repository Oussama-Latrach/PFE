import numpy as np


def check_split(file_path, name):
    data = np.load(file_path)
    print(f"\n📂 Fichier : {name}")
    print(f"Nombre total de points : {data.shape[0]}")

    classes, counts = np.unique(data[:, 5], return_counts=True)
    total = data.shape[0]

    print("Distribution des classes :")
    for c, count in zip(classes, counts):
        pourcentage = (count / total) * 100
        print(f"  - Classe {int(c)} : {count} points ({pourcentage:.2f}%)")


# Vérification des trois fichiers
check_split("train.npy", "train")
check_split("val.npy", "validation")
check_split("test.npy", "test")
