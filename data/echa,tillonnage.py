import numpy as np
import os
from sklearn.model_selection import train_test_split


def extract_samples(original_train_path, original_val_path, original_test_path,
                    output_dir="extracted_data",
                    train_size=7500, val_size=1500, test_size=1500):
    """
    Extrait des sous-ensembles à partir des fichiers initiaux
    Args:
        original_train_path: Chemin vers le fichier train.npy original
        original_val_path: Chemin vers le fichier val.npy original
        original_test_path: Chemin vers le fichier test.npy original
        output_dir: Dossier de sortie pour les nouveaux fichiers
        train_size: Nombre de points pour l'entraînement
        val_size: Nombre de points pour validation
        test_size: Nombre de points pour test
    """

    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)

    # Chargement des données originales
    train_data = np.load(original_train_path)
    val_data = np.load(original_val_path)
    test_data = np.load(original_test_path)

    print(f"Données originales - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")

    # Fusionner toutes les données pour une répartition propre
    all_data = np.concatenate([train_data, val_data, test_data])
    np.random.shuffle(all_data)  # Mélanger les données

    # Séparation en sous-ensembles
    train = all_data[:train_size]
    val = all_data[train_size:train_size + val_size]
    test = all_data[train_size + val_size:train_size + val_size + test_size]

    # Vérification des tailles
    print(f"Nouvelles tailles - Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")

    # Vérification de la répartition des classes
    def check_distribution(data, name):
        unique, counts = np.unique(data[:, 5], return_counts=True)
        print(f"\nDistribution des classes ({name}):")
        for cls, count in zip(unique, counts):
            print(f"Classe {int(cls)}: {count} points ({count / len(data):.2%})")

    check_distribution(train, "Train")
    check_distribution(val, "Validation")
    check_distribution(test, "Test")

    # Sauvegarde des nouveaux fichiers
    np.save(os.path.join(output_dir, "train_extracted.npy"), train)
    np.save(os.path.join(output_dir, "val_extracted.npy"), val)
    np.save(os.path.join(output_dir, "test_extracted.npy"), test)

    print("\nExtraction terminée avec succès!")


# Exemple d'utilisation
if __name__ == "__main__":
    # Chemins vers vos fichiers originaux
    original_train = "train.npy"
    original_val = "val.npy"
    original_test = "test.npy"

    # Appel de la fonction
    extract_samples(original_train, original_val, original_test,
                    train_size=7500, val_size=1500, test_size=1500)