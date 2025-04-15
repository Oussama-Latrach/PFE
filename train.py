import torch
import torch.optim as optim
import torch.nn as nn
from dgcnn import DGCNN
from data_loader import get_loaders
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    num_classes = 6
    epochs = 5
    batch_size = 32
    learning_rate = 0.001
    weight_decay = 1e-4

    # Dossiers de sauvegarde
    os.makedirs('metrics', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Data Loaders
    train_loader, val_loader, test_loader = get_loaders(batch_size)

    # Model
    model = DGCNN(num_classes=num_classes).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Loss function with class weights
    class_counts = np.array([751, 3367, 64, 168, 1699, 1451])
    class_weights = torch.tensor(1. / (class_counts / class_counts.sum()), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Métriques à sauvegarder
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    def train_epoch(model, loader):
        model.train()
        total_loss, correct = 0, 0

        for data, target in loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (output.argmax(1) == target).sum().item()

        return total_loss / len(loader), correct / len(loader.dataset)

    def evaluate(model, loader):
        model.eval()
        total_loss, correct = 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                loss = criterion(output, target)
                total_loss += loss.item()
                correct += (output.argmax(1) == target).sum().item()

                all_preds.extend(output.argmax(1).cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        print("\nClassification Report:")
        print(classification_report(all_targets, all_preds,
                                    target_names=[f"Class {i + 1}" for i in range(num_classes)],
                                    digits=4))

        return total_loss / len(loader), correct / len(loader.dataset)

    # Training
    print("\nStarting training...")
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader)
        val_loss, val_acc = evaluate(model, val_loader)
        scheduler.step()

        # Sauvegarde des métriques
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')

    # Sauvegarde finale du modèle
    torch.save(model.state_dict(), 'models/final_model.pth')

    # Sauvegarde des métriques
    np.save('metrics/training_metrics.npy', metrics)

    # Visualisation des courbes
    plt.figure(figsize=(12, 5))

    # Courbe de loss
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Courbe d'accuracy
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_acc'], label='Train Accuracy')
    plt.plot(metrics['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('metrics/training_curves.png')
    plt.close()

    # Final Evaluation
    print("\nTesting best model...")
    model.load_state_dict(torch.load('models/best_model.pth'))
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%}")


if __name__ == '__main__':
    main()