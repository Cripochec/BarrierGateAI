from ultralytics import YOLO
import cv2

# Загружаем предобученную модель YOLOv8
model = YOLO("yolov8n.pt")  # Можно взять yolov8s или yolov8m для большей точности

# Загружаем изображение
img_path = "car.jpg"
img = cv2.imread(img_path)

# Детектим объекты
results = model(img)[0]  # первый кадр (т.к. возвращается batch)

# Проходим по найденным объектам
for box in results.boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    if results.names[cls] in ['car', 'truck', 'bus']:  # фильтруем только автомобили
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = img[y1:y2, x1:x2]
        cv2.imwrite("Ml_color/car.jpg", cropped)
        print(f"Сохранил обрезанный фрагмент машины: {x1}, {y1}, {x2}, {y2}")
