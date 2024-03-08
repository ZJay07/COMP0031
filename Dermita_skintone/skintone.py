from derm_ita import get_ita
from PIL import Image
from derm_ita import get_fitzpatrick_type

import cv2

'''
The 'derm_ita' library is sometimes not accurate with white background so first crop faces. This will improve accuracy

ref: https://www.geeksforgeeks.org/cropping-faces-from-images-using-opencv-python/

'''

# Image cropping to face region:

name = "./images/test_images/test_1.jpeg"
# Read the input image 
img = cv2.imread(name) 

# Convert into grayscale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

# Load the cascade 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces 
faces = face_cascade.detectMultiScale(gray, 1.1, 4) 

# Draw rectangle around the faces and crop the faces - faces detected is saved in cropped folder
# Note for the image currently soecified it incorrectly identifies 2 other faces. But the first one identified is usually the correct one.
counter = 1
for (x, y, w, h) in faces:
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2) 
	faces = img[y:y + h, x:x + w] 
	cv2.imshow("face",faces) 
	# save cropped image to whatever path
	cv2.imwrite(f'./images/cropped/test_1_cropped_{counter}.jpg', faces) 	
	counter+=1

# Display and save the output the output 
cv2.imwrite('./images/output/detected.jpg', img) 
cv2.imshow('img', img) 


'''
NOTE: for some reason this doesn't work with PNG images, but works fine with JPG/JPEG images. I haven't tested it on any other types.

See link below for library installation
ref: https://github.com/AdamCorbinFAUPhD/derm_ita

'''

# Get fitzpack skintone type from cropped face

img_path = "./images/cropped/test_1_cropped_1.jpg"
img = Image.open(img_path)
whole_image_ita = get_ita(img)

type = get_fitzpatrick_type(whole_image_ita)

print(type)
