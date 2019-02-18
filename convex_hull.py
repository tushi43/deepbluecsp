import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread('images/p2.jpg')
blur = cv2.blur(image,(5,5))
plt.subplot(121),plt.imshow(image),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

img = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#
ret,thresh_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# thresh_img = cv2.Canny(thresh_img,100,200)
# cv2.imshow("Edges",thresh_img)
# ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# ret,thresh_img = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
# ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
#
# titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
#
# for i in range(6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()

contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.imshow('image',im2)
hull = []

for i in range(len(contours)):
    hull.append(cv2.convexHull(contours[i], False))

# print(contours)
drawing = np.zeros((thresh_img.shape[0], thresh_img.shape[1], 3), np.uint8)

for i in range(len(contours)):
    color_contours = (0, 255, 0)  # green - color for contours
    color = (255, 0, 255)  # blue - color for convex hull
    # rect = cv2.minAreaRect(contours[i])
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
    cv2.drawContours(drawing, hull, i, color, 1, 8)
# thresh = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
alpha = 1
beta = 1
thresh = cv2.addWeighted(drawing, alpha, image, beta, 0.2)
cv2.imshow("Output", thresh)
# cv2.imshow("Output Real", drawing)