import cv2 as cv
import numpy as np

image_path = r"finaltest2\1\000000033_077515.jpg"

image = cv.imread(image_path)
img_resized = cv.resize(image, (100,30))  #調整圖片縮放大小

gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)


cv.imshow('Result', gray)
cv.waitKey(0)
