import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread('foo.png')
blur = cv2.blur(image,(5,5))
# blur = cv2.blur(blur,(5,5))
# blur = cv2.medianBlur(image,5)
blur = cv2.medianBlur(blur,5)
# blur = cv2.medianBlur(blur,5)
# blur = cv2.bilateralFilter(blur, 9, 75, 75)
blur = cv2.bilateralFilter(blur, 9, 75, 75)
blur = cv2.bilateralFilter(blur, 9, 75, 75)
blur = cv2.bilateralFilter(blur, 9, 75, 75)
blur = cv2.bilateralFilter(blur, 9, 75, 75)
blur = cv2.bilateralFilter(blur, 9, 75, 75)
# blur = cv2.blur(blur,(5,5))
plt.subplot(121),plt.imshow(image),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
img = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

ret,thresh_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY)
# thresh1 = cv2.Canny(thresh_img,100,200)
# cv2.imshow("Edges",thresh_img)
# ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
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
fgbg = cv2.createBackgroundSubtractorMOG2(200,cv2.THRESH_BINARY,1)
masked_image = fgbg.apply(thresh_img)
# cv2.imshow('im1',masked_image)
# masked_image = fgbg.apply(thresh1)
# cv2.imshow('im2',masked_image)
# masked_image = fgbg.apply(thresh2)
# cv2.imshow('im3',masked_image)
# masked_image = fgbg.apply(thresh3)
# cv2.imshow('im4',masked_image)
# masked_image = fgbg.apply(thresh4)
# cv2.imshow('im5',masked_image)
# masked_image = fgbg.apply(thresh5)
# cv2.imshow('im6',masked_image)
# contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # cv2.imshow('image',im2)
# hull = []
#
# for i in range(len(contours)):
#     hull.append(cv2.convexHull(contours[i], False))
#
# # print(contours)
# drawing = np.zeros((thresh_img.shape[0], thresh_img.shape[1], 3), np.uint8)
#
# for i in range(len(contours)):
#     color_contours = (0, 255, 0)  # green - color for contours
#     color = (255, 0, 0)  # yellow - color for convex hull
#     rect = cv2.minAreaRect(contours[i])
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)
#     cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
#     # cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
#     # cv2.drawContours(drawing, hull, i, color, 1, 8)
# # thresh = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
# alpha = 1
# beta = 1
# thresh = cv2.addWeighted(drawing, alpha, image, beta, 0)
# cv2.imshow("Output", thresh)
# # cv2.imshow("Output Real", drawing)
kernel = np.ones((7,7),np.uint8)
dilated_img = cv2.dilate(masked_image,kernel,iterations = 1)
# cv2.imshow('Dilation',dilated_img)
eroded_img = cv2.erode(dilated_img,kernel,iterations = 1)
# cv2.imshow('Erosion',eroded_img)
# blur1 = cv2.blur(eroded_img,(5,5))

plt.imsave("foo1.png", eroded_img , format="png", cmap="gray")
drawing = np.zeros((eroded_img.shape[0], eroded_img.shape[1], 3), np.uint8)
contours, hierarchy = cv2.findContours(eroded_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hull = []
for i in range(len(contours)):
    hull.append(cv2.convexHull(contours[i], False))
for i in range(len(contours)):
    color_contours = (255, 255, 0)  # green - color for contours
    color = (255, 0, 0)  # yellow - color for convex hull
    rect = cv2.minAreaRect(contours[i])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(eroded_img, [box], 0, (0, 0, 0), 2)
# alpha = 0.5
# beta = 0.5
# thresh = cv2.addWeighted(drawing, alpha, image, beta, 0)
eroded_img = cv2.resize(eroded_img,(480,640))
cv2.imshow("Output", eroded_img)