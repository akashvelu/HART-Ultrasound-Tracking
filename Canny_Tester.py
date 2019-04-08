import numpy as np 
import cv2

from matplotlib import pyplot as plt


img = cv2.imread('images/2282.pgm',0)

edges = cv2.Canny(img,150,250)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.show()
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
