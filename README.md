# Edge-Detection
## Aim:
To perform edge detection using Sobel, Laplacian, and Canny edge detectors.

## Software Required:
Anaconda - Python 3.7

## Algorithm

### Step1:
Import the required packages.

### Step2:
Convert the input image to gray , to get more details and for laplcian operator, we have to convert input image to bgr format

### Step3:
Apply smoothing to reduce noise, here we are using gaussian blur

### Step4:
Perform the edge detector operation

### Step5:
Show the detected image
 
## Program:
~~~
NAME : ANISH MJ
REG NO : 212221230005
~~~

~~~
# Import the packages

import numpy as np
import cv2
import matplotlib.pyplot as plt


# Load the image, Convert to grayscale and remove noise

img = cv2.imread('city.jpeg')
cv2.imshow("original",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.GaussianBlur(gray,(3,3),0)
cv2.imshow("Gaussian blur",img)
cv2.waitKey(0)
cv2.destroyAllWindows()




# SOBEL EDGE DETECTOR


sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
plt.figure(1)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
cv2.imshow("SobelX",sobelx)
cv2.waitKey(0)
cv2.destroAllWindows()

sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
plt.figure(1)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')


cv2.imshow("sobely",sobely)
cv2.waitKey(0)
cv2.destroyAllWindows

sobel_xy=cv2.Sobel(img,cv2.CV_64F,1,1,ksize=5)
cv2.imshow('sobel_xy',sobel_xy)
cv2.waitKey(0)
cv2.destroyAllWindows()


# LAPLACIAN EDGE DETECTOR

# LAPLACIAN EDGE DETECTOR AND TO SHOW THE DETECTED IMAGE

rgb_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
laplacian_operator = cv2.Laplacian(rgb_image,cv2.CV_64F)
cv2.imshow('laplacian_operator',laplacian_operator)
cv2.waitKey(0)
cv2.destroyAllWindows()

# CANNY EDGE DETECTOR

canny_edges = cv2.Canny(img,120,150)
plt.figure(1)
cv2.imshow("Canny Edges",canny_edges)
cv2.waitKey(0)
cv2.destroyAllWindows

~~~

## Output:
### ORIGINAL IMAGE
![a](ori.png)
### SOBEL EDGE DETECTOR
![a](sobelx.png)
![a](sobely.png)
![a](sobelxy.png)


### LAPLACIAN EDGE DETECTOR
![p](laplacian.png)


### CANNY EDGE DETECTOR
![a](canny.png)

## Result:
Thus the edges are detected using Sobel, Laplacian, and Canny edge detectors.
