import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def depth_pr(i,image_shape,no_of_pixels=0):
    # take values angle_of_camera and vertical_distance from database
    aov = 65.999985751362348
        #64.59985901018617
    #62.99986399027697
    angle_of_camera1 = aov/2
    angle_of_camera2 = aov/2
    vertical_distance = 119.38 # cm Input variable
    # print("tan(x) in angle = ", math.tan(angle_of_camera/2))
    # print("tan(x) in radians = ", math.tan(math.radians(angle_of_camera/2)))
    distanceX = math.tan(math.radians(angle_of_camera1)) * vertical_distance
    distanceY = math.tan(math.radians(angle_of_camera2)) * vertical_distance
    total_distance = distanceY + distanceX
    # print("Total Distance = ", total_distance,"cm")

    # take value from database
    length_pixel = image_shape
    per_pixel = total_distance / length_pixel
    # print("measure per pixel =",per_pixel,"cm")

    # main code for depth

    depth_pixels = no_of_pixels
    depth_measure = depth_pixels * per_pixel

    print("Depth of pothole ",(i+1)," is ",depth_measure,"cm")
    return depth_measure



input_image = cv2.imread('images/p2.jpg')
# input_image = cv2.imread('dataset/poth11.jpg')
img = input_image.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
img = np.log(1 + img)

shapex, shapey = img.shape
M = 2*img.shape[0] + 1
N = 2*img.shape[1] + 1
sigma = 10
nx, ny = (M, N)
x = np.linspace(1, N, N)
y = np.linspace(1, M, M)
X, Y = np.meshgrid(x, y, sparse=False, indexing='xy')
centerX = int(np.ceil(N/2))
centerY = int(np.ceil(M/2))
gaussianNumerator = np.power((X - centerX),2) + np.power((Y - centerY),2)
power_sigma = 2 * np.power(sigma,2)
H = (-gaussianNumerator) / power_sigma
H = np.exp(H)
H = 1 - H
H = np.around(np.fft.fftshift(H),decimals=4)
m = int(np.ceil(M/2))
n = int(np.ceil(N/2))
img1 = np.pad(img, ((0,m),(0,n)), 'constant')
Ifft = np.fft.fft2(img1)
# print(Ifft)
IH = H * Ifft
IH = np.fft.ifft2(IH)
img_out = IH.real
# img_out = np.resize(img_out, (shapex,shapey))
img_out = np.around(img_out[0:shapex , 0:shapey],decimals=5)
Ihmf = np.around(np.exp(img_out),decimals=4) - 1

plt.imsave("img.png", Ihmf, vmin=0, vmax=1, format="png", cmap="gray")

# image = cv2.imread('foo.png')
ip_image = cv2.imread('img.png')
blur = cv2.blur(ip_image,(5,5))

blur = cv2.medianBlur(blur,5)
blur = cv2.bilateralFilter(blur, 9, 75, 75)
blur = cv2.bilateralFilter(blur, 9, 75, 75)
blur = cv2.bilateralFilter(blur, 9, 75, 75)
blur = cv2.bilateralFilter(blur, 9, 75, 75)
blur = cv2.bilateralFilter(blur, 9, 75, 75)
# plt.subplot(121),plt.imshow(image),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
# plt.xticks([]), plt.yticks([])
# plt.show()
img = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

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
kernel = np.ones((7,7),np.uint8)

eroded_img = masked_image
# dilated_img = cv2.dilate(masked_image,kernel,iterations = 2)
# eroded_img = cv2.erode(dilated_img,kernel,iterations = 1)

# cv2.imshow('Dilation',dilated_img)
# cv2.imshow('Erosion',eroded_img)
# blur1 = cv2.blur(eroded_img,(5,5))

plt.imsave("img.png", eroded_img , format="png", cmap="gray")

inp_image = cv2.imread('img.png')
blur = cv2.blur(inp_image,(5,5))
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

for i in range(len(hull)):
    color_contours = (0, 255, 0)  # green - color for contours
    color = (255, 0, 0)  # yellow - color for convex hull
    rect = cv2.minAreaRect(hull[i])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    # cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
    # cv2.drawContours(drawing, hull, i, color, 1, 8)
# thresh = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
# alpha = 1
# beta = 1
# thresh = cv2.addWeighted(drawing, alpha, image, beta, 0)
# temp = cv2.resize(img, (600, 700))
temp = img
# cv2.imshow("Output", temp)



# cv2.imshow("Output Real", drawing)

# max_area = 0
# for i in range(len(contours)):
#     max_area += cv2.contourArea(contours[i])
# max_area /= len(contours)
cnts = list(filter(lambda contours: cv2.contourArea(contours) > 100, contours))
# print(len(cnts))

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale = 1.5
fontColor = (255,0,0)
lineType = cv2.LINE_AA
img = np.zeros((3096,4128),dtype= np.int32)
image_shape = inp_image.shape[0]
for i in range(len(cnts)):
    rect = cv2.minAreaRect(cnts[i])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    x1 = box[0][0]
    y1 = box[0][1]
    x2 = box[1][0]
    y2 = box[1][1]
    x3 = box[2][0]
    y3 = box[2][1]
    x4 = box[3][0]
    y4 = box[3][1]
    if ((np.sqrt(np.power(x1-x2,2)+np.power(y1-y2,2))) < (np.sqrt(np.power(x1-x3,2)+np.power(y1-y3,2)))):
        d = depth_pr(i,image_shape,(np.sqrt(np.power(x1-x2,2)+np.power(y1-y2,2))))
    else:
        d = depth_pr(i,image_shape,(np.sqrt(np.power(x1-x3,2)+np.power(y1-y3,2))))

    s = " %(depth)02f  " % {'depth': d}
    cv2.putText(drawing,
                s,
                (box[3][0], box[3][1]),
                font,
                fontScale,
                fontColor,
                lineType)
    cv2.drawContours(drawing, [box], 0, (255, 0, 0), 2)


# angle90 = 90
# (h,w)= image.shape[:2]
# center = (w/2,h/2)
# scale = 0.2
# M = cv2.getRotationMatrix2D(center, angle90, scale)
# image = cv2.warpAffine(drawing, M, (w,h))
# image = cv2.resize(drawing, (720, 980))
# cv2.imshow("Output second", drawing)
plt.subplot(121),plt.imshow(input_image),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(drawing),plt.title('Output')
plt.xticks([]), plt.yticks([])
plt.show()

temp = cv2.resize(temp, (480,640))
cv2.imshow("Segmented",temp)




