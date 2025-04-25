if __name__ == '__main__':
    import os
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, models
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    # ==== Настройки ====
    BATCH_SIZE = 32
    NUM_CLASSES = 15  # 14 цветов + "остальные"
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "color_classifier.pth"
    EARLY_STOPPING_PATIENCE = 3  # Количество эпох без улучшений до остановки

    print(f"\n📦 Используемое устройство для обучения: {DEVICE}\n")

    # ==== Преобразования ====
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ==== Загрузка данных ====
    print("🔄 Загрузка датасета...")
    train_dataset = datasets.ImageFolder('dataset_color/train', transform=transform)
    val_dataset = datasets.ImageFolder('dataset_color/val', transform=transform)
    test_dataset = datasets.ImageFolder('dataset_color/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)

    class_names = train_dataset.classes

    # ==== Модель ====
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    # ==== Обучение ====
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_losses = []
    val_accuracies = []

    best_val_accuracy = 0
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        print(f"\n🟢 Эпоха {epoch + 1}/{EPOCHS}")
        model.train()
        total_loss = 0

        train_loop = tqdm(train_loader, desc="🚂 Обучение", leave=False)
        for images, labels in train_loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"🔧 Средний loss: {avg_loss:.4f}")

        # ==== Валидация ====
        model.eval()
        correct, total = 0, 0
        val_loop = tqdm(val_loader, desc="🧪 Валидация", leave=False)
        with torch.no_grad():
            for images, labels in val_loop:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        val_accuracies.append(acc)
        print(f"✅ Validation Accuracy: {acc:.2f}%")

        # ==== Early stopping ====
        if acc > best_val_accuracy:
            best_val_accuracy = acc
            epochs_without_improvement = 0
            # Сохранение модели
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
            print(f"✅ Модель сохранена в model_epoch_{epoch+1}.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"❗ Раннее завершение на эпохе {epoch + 1}. Модель не улучшается.")
                break

    # ==== Визуализация графиков ====
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Потери при обучении")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Validation Accuracy", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Точность на валидации")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Сохраняем график
    plt.savefig("training_metrics.png")
    print("📊 Графики сохранены в файл: training_metrics.png")

    # Показываем график
    plt.show()

    # ==== Предсказание на реальном изображении ====
    def predict_color(image_path):
        model.eval()
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            predicted_color = class_names[predicted.item()]

        # Визуализация
        plt.imshow(image)
        plt.title(f"Предсказанный цвет: {predicted_color}")
        plt.axis('off')
        plt.show()
        return predicted_color

    # ==== Пример запуска ====
    if os.path.exists("car.jpg"):
        color = predict_color("car.jpg")
        print(f"🚗 Цвет автомобиля: {color}")
    else:
        print("📸 Помести изображение машины в файл 'car.jpg' для предсказания.")
