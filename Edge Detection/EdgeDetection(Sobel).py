import numpy as np
import cv2

image = cv2.imread(r"/Images/Shuyuan.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

sobel_x = np.uint8(np.absolute(sobel_x))
sobel_y = np.uint8(np.absolute(sobel_y))
sobel_combine = cv2.bitwise_or(sobel_x, sobel_y)
cv2.imshow("Edge detection by Sobel", np.hstack([gray, sobel_x, sobel_y, sobel_combine]))

cv2.waitKey(0)
