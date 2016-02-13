import numpy as np
import cv2

img = cv2.imread('bird.jpg', 1)

#Create 

cv2.imshow('Bird', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
