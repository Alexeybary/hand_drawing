
# importing the libraries
import cv2
import numpy as np

i=0
while(True): 
    i+=1
# creating an array using np.full 
# 255 is code for white color
    image = cv2.circle(image, (250+i,250+i), radius=2, color=(0, 0, 255), thickness=-1)
    # displaying the image
    cv2.imshow("image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
                    break