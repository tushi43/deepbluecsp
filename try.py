import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread('foo1.png')
blur = cv2.blur(image,(5,5))
# blur = cv2.blur(blur,(5,5))
# blur = cv2.medianBlur(image,5)
# blur = cv2.medianBlur(blur,5)
blur = cv2.medianBlur(blur,5)
blur = cv2.bilateralFilter(blur, 9, 75, 75)
blur = cv2.bilateralFilter(blur, 9, 75, 75)
blur = cv2.bilateralFilter(blur, 9, 75, 75)
blur = cv2.bilateralFilter(blur, 9, 75, 75)
blur = cv2.bilateralFilter(blur, 9, 75, 75)
blur = cv2.bilateralFilter(blur, 9, 75, 75)
blur = cv2.blur(blur,(5,5))
# plt.subplot(121),plt.imshow(image),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
# plt.xticks([]), plt.yticks([])
# plt.show()
img = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

ret,thresh_img = cv2.threshold(img,110,255,cv2.THRESH_BINARY)
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
    color = (255, 0, 0)  # yellow - color for convex hull
    rect = cv2.minAreaRect(contours[i])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    # cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
    # cv2.drawContours(drawing, hull, i, color, 1, 8)
# thresh = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
# alpha = 1
# beta = 1
# thresh = cv2.addWeighted(drawing, alpha, image, beta, 0)
image = cv2.resize(image, (600, 700))
cv2.imshow("Output", image)
# cv2.imshow("Output Real", drawing)

cnts = list(filter(lambda contours: cv2.contourArea(contours) > 100, contours))
print(len(cnts))
#print(cnts)
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale = 2.5
fontColor = (255,255,0)
lineType = cv2.LINE_AA
img = np.zeros((3096,4128),dtype= np.int32)
for i in range(len(cnts)):
    rect = cv2.minAreaRect(cnts[i])
    box = cv2.boxPoints(rect)
    for p in box:
        pt = (p[0], p[1])
        s = " %(p[0])d  %(p[1])d" %{'p[0]': p[0], "p[1]": p[1]}
        cv2.putText(drawing,
                    s,
                    pt,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
    # print('length', (np.sqrt(np.power(x4-x3,2)+np.power(y4-y3,2))))
    # print('Length of pixels', y4-y3)



for i in range(len(cnts)):
    rect = cv2.minAreaRect(cnts[i])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(drawing, [box], 0, (255, 255, 255), 2)


# angle90 = 90
# (h,w)= image.shape[:2]
# center = (w/2,h/2)
# scale = 0.2
# M = cv2.getRotationMatrix2D(center, angle90, scale)
# image = cv2.warpAffine(drawing, M, (w,h))
# image = cv2.resize(drawing, (720, 980))
# cv2.imshow("Output second", drawing)
plt.imshow(drawing)
plt.show()




x1 = box[0][0]
y1 = box[0][1]
x2 = box[1][0]
y2 = box[1][1]
x3 = box[2][0]
y3 = box[2][1]
x4 = box[3][0]
y4 = box[3][1]

print('length', (np.sqrt(np.power(x4-x3,2)+np.power(y4-y3,2))))
print('Length of pixels', y4-y3)