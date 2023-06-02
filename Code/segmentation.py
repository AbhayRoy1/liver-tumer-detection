import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# This is simply converting it to grayscale and applying the Otsu's thresholding.
img = cv.imread("./Dataset/Test/img.jpg",1)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
th1 = 0
th2 = 255
ret,thresh = cv.threshold(gray,th1,th2,cv.THRESH_BINARY + cv.THRESH_OTSU)
   
    # noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
   
    # sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)
   
    # Defining accuracuy
acc = 0.01
   
    # Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,acc*dist_transform.max(),255,0)
   
    # Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)
   
    # Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
 
    # Add one to all labels so that sure background is not 0, but 1
markers = markers+1
   
    # Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv.watershed(img,markers)
img[markers == -1] = [0,0,255]
   
   
cv.imwrite("./Result/Test/segmentation.jpg",img)
cv.imshow("Result",img)
   
cv.waitKey(0)
cv.destroyAllWindows()

   