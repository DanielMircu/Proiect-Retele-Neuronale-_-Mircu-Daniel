"""
Module pentru antrenarea rețelei neuronale

Acest fișier conține funcția `train_model` care primește date brute (numpy arrays),
construiește DataLoader-e, antrenează un `SuspensionClassifier` și returnează modelul
împreună cu istoricul de antrenare.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .model import SuspensionClassifier


def train_model(X_train, y_train, epochs=30, batch_size=32, lr=0.001):
    """Antrenează modelul de clasificare.

    Pași principali:
    1. Conversia datelor numpy la tensori Torch (FloatTensor pentru X, LongTensor pentru y)
    2. Creare TensorDataset și DataLoader pentru batch-uri
    3. Împărțire train/validation (80/20)
    4. Buclă de antrenare cu optimizare și evaluare pe setul de validare

    Args:
        X_train (np.ndarray): Features de antrenare (shape: n_samples x n_features)
        y_train (np.ndarray): Label-uri de antrenare (shape: n_samples,)
        epochs (int): Număr de epoci
        batch_size (int): Dimensiunea batch-ului
        lr (float): Learning rate pentru optimizator

    Returns:
        tuple: (model, history)
            - model: Modelul antrenat (torch.nn.Module)
            - history: Dict cu liste pentru train_loss, val_loss și val_acc
    """

    # 1) Conversia la tensori Torch
    # FloatTensor pentru features și LongTensor pentru label-uri (CrossEntropyLoss așteaptă label int)
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)

    # 2) Creare dataset și split
    dataset = TensorDataset(X_tensor, y_tensor)

    # Split train/val (80/20) - folosește random_split pentru simplitate
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # DataLoader pentru iterații în batch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 3) Inițializare model, criteriu și optimizator
    input_size = X_train.shape[1]
    model = SuspensionClassifier(input_size=input_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4) Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # ---- training ----
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)  # logits
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ---- validation ----
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        # Nu calculăm gradient pentru evaluare (mai rapid și economisește memorie)
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                # Predicție: clasa cu probabilitatea maxima (argmax pe logits)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        # Medii per epocă
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total if total > 0 else 0.0

        # Salvăm în istoric pentru ploturi/analiză ulterioară
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

    return model, history