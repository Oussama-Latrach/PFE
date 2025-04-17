import torch
from sklearn.metrics import classification_report
from projet2.models.dgcnn import DGCNN
from projet2.utils.data_loader import test_loader

# Charger le modèle
model = DGCNN(num_classes=6).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.load_state_dict(torch.load('model.pth'))

# Évaluation du modèle
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        output = model(data)
        _, predicted = torch.max(output, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

# Rapport de classification
print(classification_report(all_labels, all_preds, digits=4))
