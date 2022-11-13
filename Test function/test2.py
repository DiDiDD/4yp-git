import cv2
from matplotlib import pyplot as plt

im = cv2.imread('siberian-husky.jpeg')
print(im.shape)

# plt.hist(im.ravel(), bins=50, density=True)
# plt.xlabel("pixel values")
# plt.ylabel("relative frequency")
# plt.title("distribution of pixels")
# plt.show()

norm_image = cv2.normalize(im, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
plt.imshow(norm_image)
plt.show()
print(norm_image.shape)

plt.hist(norm_image.ravel())
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("2distribution of pixels")
plt.show()