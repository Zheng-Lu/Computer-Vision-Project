# Importing all necessary libraries
import cv2
import os

# Read the video from specified path
vidcap = cv2.VideoCapture(r"C:\Users\Lenovo\Desktop\xxqg.mp4")


def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = vidcap.read()
    if hasFrames:
        cv2.imwrite("D:\\AI Projects\\Computer Vision Projects\\data\\" + "image" + str(count) + ".jpg",
                    image)  # save frame as JPG file
    return hasFrames


sec = 0
frameRate = 1  # //it will capture image in each 0.5 second
count = 1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)

# Release all space and windows once done

cv2.destroyAllWindows()
