from FaceRecognition import FaceRecognition
import cv2

model = FaceRecognition()

image_1 = model.load_image_file('Images/UncleDamo.jpg')
image_2 = model.load_image_file('Images/Shuyuan.jpg')

image_1_encoding = model.face_embeddings(image_1)[0]
image_2_encoding = model.face_embeddings(image_2)[0]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

