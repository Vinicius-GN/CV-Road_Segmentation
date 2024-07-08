import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Images/SP4.png')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv', img)
cv2.waitKey(0)

h, s, v =cv2.split(hsv_img)

cv2.imshow('h', h)
cv2.waitKey(0)
cv2.imshow('s', s)
cv2.waitKey(0)
cv2.imshow('v', v)
cv2.waitKey(0)


h_hist = plt.hist(h.ravel(), 180, [0, 180])
plt.plot()
plt.title("Hue Histogram")
plt.show()
s_hist = plt.hist(s.ravel(), 256, [0, 256])
plt.plot()
plt.title("Saturation Histogram")
plt.show()
v_hist = plt.hist(v.ravel(), 256, [0, 256])
plt.plot()
plt.title("Value Histogram")
plt.show()

