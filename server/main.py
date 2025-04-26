import cv2
from ultralytics import YOLO
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import easyocr
import numpy as np


def crop_vehicle(image_path: str, visualize, yolo_model=None) -> Image.Image or None:
    # Используем предзагруженную модель, если передана
    if yolo_model is None:
        yolo_model = YOLO("ML/yolov8n.pt")
    img_bgr = cv2.imread(image_path)
    results = yolo_model(img_bgr)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        if results.names[cls] in ['car', 'truck', 'bus']:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = img_bgr[y1:y2, x1:x2]
            cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

            if visualize:
                plt.imshow(cropped_pil)
                plt.title(f"Найден автомобиль: координаты ({x1}, {y1}, {x2}, {y2})")
                plt.axis('off')
                plt.show()

            return cropped_pil

    print("❌ Машина не найдена.")
    return None


def load_color_model(model_path, class_names, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return model, transform, device


def predict_color(pil_image: Image.Image, model, transform, class_names, device, visualize=False):
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted_color = class_names[predicted.item()]

    if visualize:
        plt.imshow(pil_image)
        plt.title(f"Предсказанный цвет: {predicted_color}")
        plt.axis('off')
        plt.show()

    return predicted_color


def predict_brand(pil_image: Image.Image, model, transform, class_names, device, visualize=False):
    """
    Предсказывает марку автомобиля по изображению с помощью обученной модели.

    :param pil_image: PIL.Image - изображение автомобиля
    :param model: torch.nn.Module - загруженная модель марки
    :param transform: torchvision.transforms - преобразования для входа
    :param class_names: list - список названий марок
    :param device: torch.device - устройство для инференса
    :param visualize: bool - визуализировать результат
    :return: str - предсказанная марка
    """
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted_brand = class_names[predicted.item()]

    if visualize:
        plt.imshow(pil_image)
        plt.title(f"Предсказанная марка: {predicted_brand}")
        plt.axis('off')
        plt.show()

    return predicted_brand


def detect_license_plate(img_pil: Image.Image, use_gpu: bool = False, visualize: bool = False, reader=None) -> str:
    """
    Распознаёт госномер на изображении автомобиля с помощью easyocr.

    :param img_pil: PIL изображение машины
    :param use_gpu: Использовать GPU (если доступно)
    :param visualize: Визуализировать результат с рамкой
    :param reader: Предзагруженный easyocr.Reader (опционально)
    :return: Распознанный номер или 'номер не найден'
    """
    allowed_chars = 'ABEKMHOPCTYX0123456789'
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=use_gpu)
    img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    results = reader.readtext(img_np, allowlist=allowed_chars)

    for bbox, text, conf in results:
        clean_text = ''.join(char for char in text if char in allowed_chars)
        if 6 <= len(clean_text) <= 10 and conf > 0.5:
            print(f"📷 Госномер: {clean_text} (уверенность: {conf:.2f})")

            if visualize:
                pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(img_np, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(img_np, clean_text, (pts[0][0], pts[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Отображение с matplotlib
                plt.imshow(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
                plt.title(f"Распознан номер: {clean_text}")
                plt.axis('off')
                plt.show()

            return clean_text

    if visualize:
        plt.imshow(img_pil)
        plt.title("Госномер не найден")
        plt.axis('off')
        plt.show()

    return "номер не найден"
