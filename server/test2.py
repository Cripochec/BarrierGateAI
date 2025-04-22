import cv2
# import numpy as np
import pytesseract
# import easyocr

# Установите путь к исполняемому файлу Tesseract OCR, если он отличается от значения по умолчанию.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread("../pythonProject/image.jpg")

# Перевод в чб
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plate = cv2.CascadeClassifier('../pythonProject/plate.xml')

res = plate.detectMultiScale(gray, scaleFactor=2, minNeighbors=1)

for (x, y, w, h) in res:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
    plate_image = img[y:y+h, x:x+w]
    plate_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    # text = pytesseract.image_to_string(gray)
    # reader = easyocr.Reader(['en'])
    # text = reader.readtext(plate_gray)


cv2.imshow('res', img)
cv2.waitKey(0)
# print(text)

