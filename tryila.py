import cv2
import numpy as np
from matplotlib import pyplot as plt
im = cv2.imread('images/p2.jpg')
# im = cv2.imread('dataset/pot11.jpg')
morph = im.copy()
for i in range(3):
    morph = cv2.bilateralFilter(morph,9,75,75)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

# take morphological gradient
gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)

# split the gradient image into channels
image_channels = np.split(np.asarray(gradient_image), 3, axis=2)

channel_height, channel_width, _ = image_channels[0].shape

# apply Otsu threshold to each channel
for i in range(0, 3):
    _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))

# merge the channels
image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)
# image_channels = cv2.resize(image_channels,(480,640))

cv2.imshow("Ouytpot",image_channels)
# save the denoised image

# cv2.imwrite('output.jpg', image_channels)

bw_img = cv2.cvtColor(image_channels, cv2.COLOR_BGR2GRAY)

for i in range(10):
    #bw_img = cv2.medianBlur(bw_img,5)
    bw_img = cv2.bilateralFilter(bw_img,9,75,75)

# equ = cv2.equalizeHist(bw_img)
# res = np.hstack((bw_img,equ)) #stacking images side-by-side
# cv2.imwrite('res.png',res)
# plt.hist(equ.ravel(),256,[0,256])
# plt.show()
# equalized = [val[0] for val in equ]
# indices = list(range(0, 256))
# s = [(x,y) for y,x in sorted(zip(equalized,indices), reverse=True)]
# index_of_highest_peak = s[0][0]
# index_of_second_highest_peak = s[127][0]
# index_of_highest_peak = index_of_highest_peak + index_of_second_highest_peak
# index_of_highest_peak /= 2
# print(index_of_highest_peak)
# print(index_of_second_highest_peak)
# bw_img = cv2.fastNlMeansDenoising(bw_img,None,10,10,7,21)
ret,thresh_img = cv2.threshold(bw_img,110,255,cv2.THRESH_BINARY)
temp = cv2.resize(thresh_img, (774,1032))
cv2.imshow("threshold",temp)
thresh_img = cv2.Canny(thresh_img,100,200)




contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

hull = []

for i in range(len(contours)):
    hull.append(cv2.convexHull(contours[i], False))

drawing = np.zeros((thresh_img.shape[0], thresh_img.shape[1], 3), np.uint8)

for i in range(len(hull)):
    color_contours = (0, 255, 0)  # green - color for contours
    color = (255, 0, 0)  # yellow - color for convex hull
    rect = cv2.minAreaRect(hull[i])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(im, [box], 0, (0, 255, 0), 2)
    # cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
    # cv2.drawContours(drawing, hull, i, color, 1, 8)
# thresh = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
alpha = 1
beta = 1
thresh = cv2.addWeighted(drawing, alpha, im, beta, 0)
thresh = cv2.resize(thresh, (480,640))
cv2.imshow("Output", thresh)
# cv2.imshow("Output Real", drawing)
