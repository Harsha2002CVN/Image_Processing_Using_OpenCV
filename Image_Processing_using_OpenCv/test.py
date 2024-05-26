import cv2 as cv 
import numpy as np 
from rembg import remove


img2 = cv.imread(r"input_2.jpg")
h, w, _ = img2.shape

img1 = cv.imread(r"input_1.jpg")
img1 = cv.resize(img1, (w, h))

# Removing the background from the car image 
rembg_out = remove(img2)
np_array = np.array(rembg_out)
cv_car_img = cv.cvtColor(np_array,cv.COLOR_RGBA2RGB)


# Create a mask for img2
img2gray = cv.cvtColor(np_array, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(img2gray, 30, 255, cv.THRESH_BINARY_INV)
# # Apply morphological transformations to the mask
kernel = np.ones((70,70), np.uint8)  
mask = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel=kernel)


bit_and = cv.bitwise_and(img1,img1,mask=mask)
out = cv.add(bit_and,cv_car_img)

cv.imshow("result",out)
cv.waitKey(0)
cv.destroyAllWindows()

