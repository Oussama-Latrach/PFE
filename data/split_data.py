import numpy as np

# Charger les données
data = np.load("data.npy")
print(f"Nombre total de points : {data.shape[0]}")

# Mélanger les données
np.random.seed(42)  # Pour reproductibilité
np.random.shuffle(data)

# Définir les tailles
n_total = data.shape[0]
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val  # Le reste

# Découper
train_data = data[:n_train]
val_data = data[n_train:n_train + n_val]
test_data = data[n_train + n_val:]

# Sauvegarder
np.save("train.npy", train_data)
np.save("val.npy", val_data)
np.save("test.npy", test_data)

print("✅ Split terminé :")
print(f"  - Train : {train_data.shape[0]} points")
print(f"  - Val   : {val_data.shape[0]} points")
print(f"  - Test  : {test_data.shape[0]} points")
