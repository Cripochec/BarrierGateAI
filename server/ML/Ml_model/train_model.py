import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 1. Параметры
data_dir = 'dataset_model'
batch_size = 8
num_epochs = 50
patience = 5  # для ранней остановки
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Преобразования
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 3. Загрузка данных
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 4. Модель
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model = model.to(device)

# 5. Оптимизатор и лосс
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Для графиков
train_losses = []
val_losses = []
val_accuracies = []

# Параметры ранней остановки
best_val_loss = float('inf')
patience_counter = 0

# 6. Обучение
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{num_epochs} [Обучение]")

    for images, labels in train_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.set_postfix({'loss': running_loss / (train_bar.n + 1)})

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Валидация
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    val_bar = tqdm(val_loader, desc=f"Эпоха {epoch+1}/{num_epochs} [Валидация]")

    with torch.no_grad():
        for images, labels in val_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            val_bar.set_postfix({'val_loss': val_loss / (val_bar.n + 1)})

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = correct / total
    val_accuracies.append(val_accuracy)

    print(f"\nЭпоха {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Accuracy={val_accuracy:.4f}")

    # Сохранение модели после каждой эпохи
    model_save_path = f'model_epoch_{epoch+1}.pth'
    torch.save(model.state_dict(), model_save_path)

    # Ранняя остановка
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Ранняя остановка!")
            break

# 7. Построение графиков
plt.figure(figsize=(12, 5))

# График потерь
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

# График точности
plt.subplot(1, 2, 2)
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy')

plt.tight_layout()
plt.savefig('training_stats.png')
plt.show()
