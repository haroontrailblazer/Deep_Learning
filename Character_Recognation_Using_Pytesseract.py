import cv2
import pytesseract
import numpy as np

img = cv2.imread(r"File_path")
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thr = cv2.threshold(grey, 150, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("Thresholded Image", thr)
cv2.waitKey(0)
cv2.destroyAllWindows()