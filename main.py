








import cv2
import numpy as np
# import easyocr

# img = cv2.imread("image.jpg")
cap = cv2.VideoCapture(0)
# text = easyocr.Reader(['en'])

while True:
    success, img = cap.read()

    # Перевод в чб
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plate = cv2.CascadeClassifier('../pythonProject/plate.xml')

    res = plate.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=1)

    for (x, y, w, h) in res:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
        plate_image = img[y:y+h, x:x+w]
        # text = text.readtext(plate_image)
        # print(text)





    # img = cv2.bitwise_and(img, img, mask=res)

    cv2.imshow('res', plate_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
