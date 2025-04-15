import laspy
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Fonction pour remapper les classes non intéressantes vers la classe 1 (Unclassified)
def remap_classification(value, classes_cibles):
    return value if value in classes_cibles else 1  # Remplacer les autres par la classe 1

# Chargement du fichier LAZ
fichier_laz = "data/LHD_FXX_0645_6859_PTS_O_LAMB93_IGN69.copc.laz"  # Remplace par ton fichier
las = laspy.read(fichier_laz)

# Extraire les attributs (x, y, z, ReturnNumber, NumberOfReturns, Classification)
x = las.x
y = las.y
z = las.z
rn = las.return_number
nr = las.number_of_returns
cls = las.classification

# Filtrage des classes intéressantes (classes 1, 2, 3, 4, 5, 6) et affectation des autres à 1 (Unclassified)
classes_interessees = [1, 2, 3, 4, 5, 6]  # Les classes que tu veux garder
vectorized_remap = np.vectorize(lambda x: remap_classification(x, classes_interessees))
cls_remapped = vectorized_remap(cls)

# Normalisation des coordonnées (x, y, z) entre 0 et 1
scaler = MinMaxScaler()
xyz = np.column_stack((x, y, z))  # Concaténer x, y, z
xyz_normalized = scaler.fit_transform(xyz)

# Combinaison des attributs (x, y, z, ReturnNumber, NumberOfReturns, classification)
data = np.column_stack((xyz_normalized, rn, nr, cls_remapped))

# Sauvegarder les données dans un fichier .npy
np.save("data.npy", data)

# Afficher un aperçu des 5 premières lignes du fichier de données
print("Extrait des 5 premières lignes des données traitées :")
print(data[:5])

# Statistiques des classes après remappage
from collections import Counter
compte = Counter(cls_remapped)
print("\nDistribution des classes après remappage (classe 1 = Unclassified) :")
for k, v in sorted(compte.items()):
    print(f"  - Classe {k}: {v} points")
