import time
from time import strftime
from time import gmtime
import cv2
import os

start = time.time()

def img_similarity(img1_path, img2_path):
    """
    :param img1_path: path of image1
    :param img2_path: path of image2
    :return: Similarity
    """
    try:
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)  # trainning picture
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        w2, h2 = img2.shape
        img1 = cv2.resize(img1, (h2, w2))
        # cv2.imshow("image1", img1)
        # cv2.imshow("image2", img2)
        cv2.waitKey(0)

        # Initiate ORB detector
        orb = cv2.ORB_create()

        # find the keypoint(kp) with ORB, and compute the descriptor(des)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # extract and calculate the keypoint
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        # knn sorting result
        matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, img2, flags=2)
        # cv2.imshow("img3:", img3)
        # cv2.waitKey(0)

        # 查看最大匹配点数目
        good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
        similarity = float(len(good)) / len(matches)
        print(f"Similarity:{similarity}")
        return similarity

    except:
        print('Cannot calculate the similarity')


if __name__ == '__main__':
    image1 = r"C:\Users\Lenovo\Desktop\pics\326992_1.jpg"
    image2 = r"C:\Users\Lenovo\Desktop\pics\81KXAlq0okL._AC_SX569_.jpg"
    similarity = img_similarity(image1, image2)

end = time.time()
seconds = round(end - start, 2)
if seconds < 60:
    print('Time taken: ' + str(seconds) + 's')
else:
    print('Time taken: ' + strftime("%H:%M:%S", gmtime(seconds)))