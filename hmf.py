import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('dataset/base images/poth19.jpg')
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

# I = cv2.imread('Images/ahh.png')
# I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
# print(I)
# II = I - Ihmf
# plt.imshow(I)
# plt.show()

# plt.imshow(Ihmf)
# plt.imshow(Ihmf, cmap='gray', vmin=0, vmax=1)
# img = plt.imshow(Ihmf, vmin=0, vmax=1 )
# img.set_cmap('gray')
# plt.axis('off')
plt.imsave("foo.png", Ihmf, vmin=0, vmax=1, format="png", cmap="gray")
# plt.savefig("test.png", bbox_inches='tight', pad_inches = 0)
# shadow extraction

# Ihmf = np.around(Ihmf[200:shapex , 200:shapey],decimals=5)
# plt.imshow(Ihmf, cmap='gray', vmin=0, vmax=1)
# blur = cv2.blur(Ihmf,(5,5))
# blur = cv2.blur(blur,(5,5))
# blur = cv2.blur(blur,(5,5))
# blur = cv2.medianBlur(blur,5)
# blur = cv2.medianBlur(blur,5)
# blur = cv2.bilateralFilter(blur, 9, 75, 75)
# blur = cv2.bilateralFilter(blur, 9, 75, 75)
# ret,thresh_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY)
# fgbg = cv2.createBackgroundSubtractorMOG2(250,cv2.THRESH_BINARY,1)
# masked_image = fgbg.apply(thresh_img)
# cv2.imshow('im1',masked_image)
