import os
import time
from datetime import datetime
import keyboard  # pip install keyboard
import pyautogui  # pip install pyautogui
from main import crop_vehicle

# Папка для сохранения результатов
SAVE_DIR = "dataset_model/lada_vesta_rest"
os.makedirs(SAVE_DIR, exist_ok=True)

# Клавиша для активации (например, F9)
HOTKEY = "F9"


def main():
    n=0
    print(f"Скрипт запущен. Для скриншота и обработки нажмите {HOTKEY}...")
    while True:
        keyboard.wait(HOTKEY)
        print("⌨️ Кнопка нажата, делаю скриншот...")
        screenshot = pyautogui.screenshot()
        temp_path = os.path.join(SAVE_DIR, "temp_screen.png")
        screenshot.save(temp_path)

        # Обработка скриншота
        cropped = crop_vehicle(temp_path, visualize=False)
        if cropped:
            n += 1
            save_path = os.path.join(SAVE_DIR, f"car_{n}.png")
            cropped.save(save_path)
            print(f"✅ Авто найдено и сохранено: {save_path}")
        else:
            print("❌ Авто не найдено на скриншоте.")

        # Удаляем временный скриншот
        if os.path.exists(temp_path):
            os.remove(temp_path)
        time.sleep(0.5)  # чтобы не было двойных срабатываний

if __name__ == "__main__":
    main()