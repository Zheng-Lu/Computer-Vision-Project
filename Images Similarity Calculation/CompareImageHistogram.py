import time
from time import strftime
from time import gmtime
import cv2


def compare_img_hist(img1, img2):
    """
    Compare the similarity of two pictures using histogram(直方图)
        Attention: this is a comparision of similarity, using histogram to calculate
        For example:
         1. img1 and img2 are both 720P .PNG file,
            and if compare with img1, img2 only add a black dot(about 9*9px),
            the result will be 0.999999999953
    :param img1: img1 in MAT format(img1 = cv2.imread(image1))
    :param img2: img2 in MAT format(img2 = cv2.imread(image2))
    :return: the similarity of two pictures
    """
    # Get the histogram data of image 1, then using normalize the picture for better compare
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    img1_hist = cv2.calcHist([img1], [1], None, [256], [0, 256])
    img1_hist = cv2.normalize(img1_hist, img1_hist, 0, 1, cv2.NORM_MINMAX, -1)

    img2_hist = cv2.calcHist([img2], [1], None, [256], [0, 256])
    img2_hist = cv2.normalize(img2_hist, img2_hist, 0, 1, cv2.NORM_MINMAX, -1)

    similarity = cv2.compareHist(img1_hist, img2_hist, 0)

    return similarity


start = time.time()

if __name__ == '__main__':
    image1 = r"C:\Users\Lenovo\Desktop\pics\326992_1.jpg"
    image2 = r"C:\Users\Lenovo\Desktop\pics\81KXAlq0okL._AC_SX569_.jpg"
    print(compare_img_hist(image1, image2))

end = time.time()
seconds = round(end - start, 2)
if seconds < 60:
    print('Time taken: ' + str(seconds) + 's')
else:
    print('Time taken: ' + strftime("%H:%M:%S", gmtime(seconds)))
