# import cv2
# import numpy as np
#
# img = cv2.imread("image.jpg")
#
# # new_img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
#
# # Размытие
# new_img = cv2.GaussianBlur(img, (3, 3), 3)
#
# # Перевод в серый
# new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
#
# # Размер
# print(new_img.shape)
#
# # образы
# new_img = cv2.Canny(new_img, 100, 100)
#
# # ядро
# kernal = np.ones((5, 5), np.uint8)
#
# # # Обводка увеличение
# # new_img = cv2.dilate(new_img, kernal, iterations=1)
# #
# # # Обводка уменьшение
# # new_img = cv2.erode(new_img, kernal, iterations=1)
#
# # cv2.imshow('res', new_img[0:100, 0:150])
# # cv2.imshow('res', new_img)
#
#
# # фигуры
# photo = np.zeros((300, 300, 3), dtype='uint8')
#
# # фон в BGR
# # photo[:] = 255, 100, 100
#
# # квадрат с обводкой
# cv2.rectangle(photo, (50, 50), (100, 100), (255, 0, 0), thickness=2)
#
# # Линия
# cv2.line(photo, (10, 10), (110, 10), (0, 255, 0), thickness=2)
#
# # Круг
# cv2.circle(photo, (200, 200), 50, (0, 255, 0), thickness=2)
#
# # Текст
# cv2.putText(photo, "Text", (photo.shape[0] // 2, photo.shape[1] // 2), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), )
#
# # cv2.imshow('res', photo)
#
#
#
#
# cv2.waitKey(0)
#
#
# # Функция поворота
# def rotate(imges, angle):
#     height, width = imges.shape[:2]
#     point = (width // 2, height // 2)
#
#     mat = cv2.getRotationMatrix2D(point, angle, 1)
#     return cv2.warpAffine(imges, mat, (width, height))
#
#
# # Функция среза
# def transform(img_param, x, y):
#     mat = np.float32([[1, 0, x], [0, 1, y]])
#     return cv2.warpAffine(img_param, mat, (img_param.shape[1], img_param.shape[0]))
#
#
# # cap = cv2.VideoCapture("video.mp4")
# cap = cv2.VideoCapture(0)
# # cap.set(3, 800)
# # cap.set(4, 600)
#
# while True:
#     success, img = cap.read()
#
#     # Размытие
#     img = cv2.GaussianBlur(img, (5, 5), 0)
#
#     # Перевод в серый
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Формат hsv
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     # Формат lab
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#
#     # Формат rgb
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     r, g, b = cv2.split(img)
#
#     img = cv2.merge([b, g, r])
#
#
#     # образы
#     # img = cv2.Canny(img, 40, 60)
#
#     # Контуры
#     # con, hir = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#
#     # Отзеркаливание
#     # img = cv2.flip(img, 1)
#
#     # функция поворота
#     # img = rotate(img, 35)
#
#
#
#     # print(con)
#
#
#     cv2.imshow('res', img)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#
