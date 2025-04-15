import numpy as np
import torch
import open3d as o3d
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


class DGCNNInference:
    def __init__(self, model_path):
        # Configuration du device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Définition du modèle (identique à dgcnn.py mais auto-contenu)
        class DGCNN(torch.nn.Module):
            def __init__(self, num_classes=6, k=20):
                super().__init__()
                self.k = k
                self.conv1 = torch.nn.Sequential(
                    torch.nn.Conv2d(5 * 2, 64, kernel_size=1, bias=False),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU(negative_slope=0.2))
                self.conv2 = torch.nn.Sequential(
                    torch.nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU(negative_slope=0.2))
                self.conv3 = torch.nn.Sequential(
                    torch.nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(negative_slope=0.2))
                self.conv4 = torch.nn.Sequential(
                    torch.nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.LeakyReLU(negative_slope=0.2))
                self.conv5 = torch.nn.Sequential(
                    torch.nn.Conv1d(256, 1024, kernel_size=1, bias=False),
                    torch.nn.BatchNorm1d(1024),
                    torch.nn.LeakyReLU(negative_slope=0.2))
                self.fc1 = torch.nn.Linear(1024, 512)
                self.fc2 = torch.nn.Linear(512, 256)
                self.fc3 = torch.nn.Linear(256, num_classes)
                self.dropout = torch.nn.Dropout(p=0.5)

            def forward(self, x):
                batch_size = x.size(0)
                x = x.unsqueeze(1).permute(0, 2, 1)
                x = self._edge_conv_blocks(x)
                x = self._classification(x)
                return x

            def _edge_conv_blocks(self, x):
                x1 = self._get_graph_feature(x)
                x1 = self.conv1(x1).max(dim=-1, keepdim=False)[0]
                x2 = self._get_graph_feature(x1)
                x2 = self.conv2(x2).max(dim=-1, keepdim=False)[0]
                x3 = self._get_graph_feature(x2)
                x3 = self.conv3(x3).max(dim=-1, keepdim=False)[0]
                x4 = self._get_graph_feature(x3)
                x4 = self.conv4(x4).max(dim=-1, keepdim=False)[0]
                x5 = self.conv5(x4).max(dim=-1, keepdim=False)[0]
                return x5

            def _get_graph_feature(self, x):
                batch_size, num_dims, num_points = x.size()
                if num_points == 1:
                    x = x.repeat(1, 1, self.k)
                    num_points = self.k
                x_t = x.permute(0, 2, 1)
                pairwise_distance = -torch.sum(x ** 2, dim=1, keepdim=True) - 2 * torch.matmul(x_t, x) - torch.sum(
                    x ** 2, dim=1, keepdim=True).permute(0, 2, 1)
                k = min(self.k, num_points - 1)
                idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][..., 1:]
                idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
                idx = (idx + idx_base).view(-1)
                x = x.permute(0, 2, 1).contiguous()
                neighbors = x.view(batch_size * num_points, -1)[idx, :]
                neighbors = neighbors.view(batch_size, num_points, k, num_dims)
                x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
                return torch.cat((neighbors - x, x), dim=3).permute(0, 3, 1, 2)

            def _classification(self, x):
                x = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=0.2)
                x = self.dropout(x)
                x = torch.nn.functional.leaky_relu(self.fc2(x), negative_slope=0.2)
                x = self.dropout(x)
                return self.fc3(x)

        # Chargement du modèle
        self.model = DGCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Configuration des couleurs (une par classe)
        self.colors = np.array([
            [1, 0, 0],  # Classe 1 - Rouge
            [0, 1, 0],  # Classe 2 - Vert
            [0, 0, 1],  # Classe 3 - Bleu
            [1, 1, 0],  # Classe 4 - Jaune
            [1, 0, 1],  # Classe 5 - Magenta
            [0, 1, 1]  # Classe 6 - Cyan
        ])

    def predict(self, input_npy):
        # Chargement des données
        data = np.load(input_npy)
        assert data.shape[1] == 5, "Le fichier doit avoir exactement 5 colonnes"

        # Conversion en tensor
        points = torch.tensor(data, dtype=torch.float32).to(self.device)

        # Prédiction par batch
        predictions = []
        batch_size = 1024  # Peut être ajusté selon la mémoire disponible

        with torch.no_grad():
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                pred = self.model(batch)
                predictions.append(pred.argmax(dim=1).cpu().numpy())

        # Concaténation des résultats
        class_ids = np.concatenate(predictions)

        # Ajout des classes aux données originales
        classified_data = np.column_stack((data, class_ids + 1))  # +1 pour revenir aux classes 1-6

        return classified_data

    def visualize(self, classified_data):
        # Création du nuage de points Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(classified_data[:, :3])  # XYZ

        # Attribution des couleurs selon les classes
        colors = self.colors[classified_data[:, 5].astype(int) - 1]  # -1 car les classes sont 1-6
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Visualisation
        o3d.visualization.draw_geometries([pcd],
                                          window_name="Prédiction DGCNN",
                                          width=1024,
                                          height=768,
                                          point_show_normal=False)

        # Optionnel: Sauvegarde du résultat
        output_file = input_npy.replace('.npy', '_classified.ply')
        o3d.io.write_point_cloud(output_file, pcd)
        print(f"Résultats sauvegardés dans {output_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Inférence DGCNN pour classification de nuages de points')
    parser.add_argument('input_file', type=str, help='Chemin vers le fichier .npy d\'entrée (5 colonnes)')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                        help='Chemin vers le modèle entraîné (default: models/best_model.pth)')

    args = parser.parse_args()

    # Initialisation et prédiction
    inferencer = DGCNNInference(args.model)
    result = inferencer.predict(args.input_file)

    # Visualisation
    inferencer.visualize(result)

    # Affichage des statistiques
    unique, counts = np.unique(result[:, 5], return_counts=True)
    print("\nStatistiques de classification:")
    for cls, count in zip(unique, counts):
        print(f"Classe {int(cls)}: {count} points ({count / len(result):.2%})")