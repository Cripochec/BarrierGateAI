import easyocr
import cv2

img = cv2.imread("textru.png")
reader = easyocr.Reader(['ru'])
res = reader.readtext(img)

print(res)
