import cv2
import numpy as np
from matplotlib import pyplot as plt

# input_image = cv2.imread('dataset/pot21.jpg')
input_image = cv2.imread('images/p1.png')
# input_image = cv2.imread('images/p1.png')
blur = cv2.medianBlur(input_image,5)
bw_img = cv2.medianBlur(blur,21)

# rgb_planes = cv2.split(bw_img)
#
# result_planes = []
# result_norm_planes = []
# for plane in rgb_planes:
#     dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
#     bg_img = cv2.medianBlur(dilated_img, 21)
#     diff_img = 255 - cv2.absdiff(plane, bg_img)
#     cv2.normalize(diff_img,dilated_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#     result_planes.append(diff_img)
#     result_norm_planes.append(dilated_img)
#
# result = cv2.merge(result_planes)
# result_norm = cv2.merge(result_norm_planes)
# result_norm_temp = cv2.resize(result_norm,(480,640))
# cv2.imshow("result",result)
# cv2.imshow("result Normalized",result_norm_temp)
#
# bw_img = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
#
# bw_img = result_norm

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

b,g,r = cv2.split(bw_img)           # get b,g,r
rgb_img = cv2.merge([r,g,b])     # switch it to rgb

dst = cv2.fastNlMeansDenoisingColored(rgb_img,None,10,10,7,21)

b,g,r = cv2.split(dst)           # get b,g,r
bw_img = cv2.merge([r,g,b])     # switch it to rgb



bw_img = cv2.cvtColor(bw_img, cv2.COLOR_BGR2GRAY)


for i in range(10):
    #bw_img = cv2.medianBlur(bw_img,11)
    bw_img = cv2.bilateralFilter(bw_img,9,75,75)


ret,thresh_img = cv2.threshold(bw_img,110,255,cv2.THRESH_BINARY)
temp = cv2.resize(thresh_img, (774,1032))
cv2.imshow("threshold",temp)
thresh_img = cv2.Canny(thresh_img,100,200)




contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

hull = []

for i in range(len(contours)):
    hull.append(cv2.convexHull(contours[i], False))

drawing = np.zeros((thresh_img.shape[0], thresh_img.shape[1], 3), np.uint8)

max_area = input_image.shape[1] * input_image.shape[0]
max_area *= 0.001
# print(len(contours))
cnts = list(filter(lambda hull: cv2.contourArea(hull) > max_area, hull))
# print(len(cnts))
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 2
fontColor = (0,255,0)
lineType = cv2.LINE_AA
img = np.zeros((3096,4128),dtype= np.int32)


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
    cv2.drawContours(input_image, [box], 0, (0, 0, 255), 5)
alpha = 1
beta = 1
thresh = cv2.addWeighted(drawing, alpha, input_image, beta, 0)
thresh = cv2.resize(thresh, (480,640))
cv2.imshow("Output", thresh)
# cv2.imshow("Output Real", drawing)

print("Number of potholes",len(cnts))