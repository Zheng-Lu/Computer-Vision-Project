import numpy as np
import cv2

image = cv2.imread(r"/Images/Shuyuan.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 30 and 150 is the threshold, larger than 150 is considered as edge,
# less than 30 is considered as not edge
canny = cv2.Canny(gray, 30, 150)

canny = np.uint8(np.absolute(canny))
# display two images in a figure
cv2.imshow("Edge detection by Canny", np.hstack([gray, canny]))


cv2.waitKey(0)

