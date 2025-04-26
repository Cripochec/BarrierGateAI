from json import load
from main import crop_vehicle, load_color_model, predict_color, detect_license_plate, predict_brand
import time
from ultralytics import YOLO
import easyocr
import config

if __name__ == '__main__':

    # Загрузка классов цвета
    with open(config.COLOR_CLASSES_PATH, "r", encoding="utf-8") as f:
        class_dict = load(f)
    class_names = [class_dict[str(i)] for i in range(len(class_dict))]

    # Загрузка классов брендов
    with open(config.BRAND_CLASSES_PATH, "r", encoding="utf-8") as f:
        brand_dict = load(f)
    brand_names = [brand_dict[str(i)] for i in range(len(brand_dict))]

    # Загрузка моделей
    yolo_model = YOLO(config.YOLO_MODEL_PATH)
    model, transform, device = load_color_model(
        model_path=config.COLOR_MODEL_PATH,
        class_names=class_names,
        device="cuda" if config.USE_GPU else "cpu"
    )
    # Загрузка модели бренда
    brand_model, brand_transform, brand_device = load_color_model(
        model_path=config.BRAND_MODEL_PATH,
        class_names=brand_names,
        device="cuda" if config.USE_GPU else "cpu"
    )
    ocr_reader = easyocr.Reader(config.OCR_LANGS, gpu=config.USE_GPU)

    # Вырезка авто
    start_crop = time.time()
    cropped_image = crop_vehicle("Image_for_test/2107-2.jpg", visualize=config.VISUALIZE, yolo_model=yolo_model)
    if not cropped_image:
        exit()
    end_crop = time.time()

    # Предсказание цвета
    start_pred_color = time.time()
    predicted_color = predict_color(
        pil_image=cropped_image,
        model=model,
        transform=transform,
        class_names=class_names,
        device=device,
        visualize=config.VISUALIZE
    )
    end_pred_color = time.time()

    # Предсказание бренда
    start_pred_brand = time.time()
    predicted_brand = predict_brand(
        pil_image=cropped_image,
        model=brand_model,
        transform=brand_transform,
        class_names=brand_names,
        device=brand_device,
        visualize=config.VISUALIZE
    )
    end_pred_brand = time.time()

    # Определение госномера
    start_plate = time.time()
    license_plate = detect_license_plate(
        cropped_image,
        use_gpu=config.USE_GPU,
        visualize=config.VISUALIZE,
        reader=ocr_reader
    )
    end_plate = time.time()

    print(f"⏱️ Вырезание авто заняло {end_crop - start_crop:.3f} секунд.")
    print(f"⏱️ Предсказание цвета заняло {end_pred_color - start_pred_color:.3f} секунд.")
    print(f"⏱️ Предсказание бренда заняло {end_pred_brand - start_pred_brand:.3f} секунд.")
    print(f"⏱️ Распознавание номера заняло {end_plate - start_plate:.3f} секунд.")

    print(f'\n🎨 Предсказанный цвет: {predicted_color}\n'
          f'🔢 Предсказанный Гос. номер: {license_plate}\n'
          f'🚗 Предсказанная марка и модель авто: {predicted_brand}')
