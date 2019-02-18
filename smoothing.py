import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
image = cv2.imread('images/p9.jpg')
# image = cv2.imread('dataset/pot71.jpg')
blur = cv2.blur(image,(7,7))
# blur = cv2.blur(blur,(5,5))
# blur = cv2.medianBlur(image,5)
# blur = cv2.medianBlur(blur,5)
# blur = cv2.medianBlur(blur,5)
# blur = cv2.bilateralFilter(blur, 9, 75, 75)
# blur = cv2.bilateralFilter(blur, 9, 75, 75)
# blur = cv2.bilateralFilter(blur, 9, 75, 75)
blur = cv2.bilateralFilter(blur, 9, 75, 75)
blur = cv2.bilateralFilter(blur, 9, 75, 75)
blur = cv2.bilateralFilter(blur, 9, 75, 75)
blur = cv2.blur(blur,(5,5))
plt.subplot(121),plt.imshow(image),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
#
# kernelG1 = np.array([[ 5,  5,  5],
#                      [-3,  0, -3],
#                      [-3, -3, -3]], dtype=np.float32)
# kernelG2 = np.array([[ 5,  5, -3],
#                      [ 5,  0, -3],
#                      [-3, -3, -3]], dtype=np.float32)
# kernelG3 = np.array([[ 5, -3, -3],
#                      [ 5,  0, -3],
#                      [ 5, -3, -3]], dtype=np.float32)
# kernelG4 = np.array([[-3, -3, -3],
#                      [ 5,  0, -3],
#                      [ 5,  5, -3]], dtype=np.float32)
# kernelG5 = np.array([[-3, -3, -3],
#                      [-3,  0, -3],
#                      [ 5,  5,  5]], dtype=np.float32)
# kernelG6 = np.array([[-3, -3, -3],
#                      [-3,  0,  5],
#                      [-3,  5,  5]], dtype=np.float32)
# kernelG7 = np.array([[-3, -3,  5],
#                      [-3,  0,  5],
#                      [-3, -3,  5]], dtype=np.float32)
# kernelG8 = np.array([[-3,  5,  5],
#                      [-3,  0,  5],
#                      [-3, -3, -3]], dtype=np.float32)
#
# g1 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
# g2 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG2), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
# g3 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG3), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
# g4 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG4), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
# g5 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG5), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
# g6 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG6), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
# g7 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG7), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
# g8 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG8), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
# magn = cv2.max(
#     g1, cv2.max(
#         g2, cv2.max(
#             g3, cv2.max(
#                 g4, cv2.max(
#                     g5, cv2.max(
#                         g6, cv2.max(
#                             g7, g8
#                         )
#                     )
#                 )
#             )
#         )
#     )
# )
# temp = cv2.resize(magn,(480,640))
# cv2.imshow("Kirsh",temp)
ret,thresh_img = cv2.threshold(gray,110,255,cv2.THRESH_BINARY)
# thresh_img = cv2.Canny(thresh_img,100,200)
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
#
# kernel = np.ones((7,7),np.uint8)
# thresh_img = cv2.dilate(thresh_img,kernel,iterations = 2)


temp = cv2.resize(thresh_img,(480,640))
cv2.imshow("Edges",temp)

contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# cv2.imshow('image',im2)
hull = []

for i in range(len(contours)):
    hull.append(cv2.convexHull(contours[i], False))

# print(contours)
drawing = np.zeros((thresh_img.shape[0], thresh_img.shape[1], 3), np.uint8)

#
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
#



max_area = image.shape[1] * image.shape[0]

entire_area = max_area

max_area *= 0.001
# print(len(contours))
cnts = list(filter(lambda hull: cv2.contourArea(hull) > max_area and cv2.contourArea(hull) < entire_area, hull))
# print(len(cnts))
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 2
fontColor = (0,0,255)
lineType = cv2.LINE_AA
# img = np.zeros((3096,4128),dtype= np.int32)


for i in range(len(cnts)):

    rect = cv2.minAreaRect(cnts[i])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    area = cv2.contourArea(cnts[i])
    s = " %(area)d  " % {'area': area}
    cv2.putText(drawing,
                s,
                (box[3][0],box[3][1]),
                font,
                fontScale,
                fontColor,
                lineType)
    cv2.drawContours(image, [box], 0, (0, 0, 255), 3)
alpha = 1
beta = 1
thresh = cv2.addWeighted(drawing, alpha, image, beta, 0)
thresh = cv2.resize(thresh, (480,640))
cv2.imshow("Output", thresh)
# cv2.imshow("Output Real", drawing)

print("Number of potholes",len(cnts))