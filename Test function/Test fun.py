from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from PIL import Image
import cv2
import torchvision.transforms as transforms
from torchvision import transforms
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor()
])

img_path = "siberian-husky.jpeg"
img = Image.open(img_path)
img_np = np.array(img,dtype=np.float32)
plt.show

img_tr = transform(img)
mean, std = img_tr.mean([1,2]), img_tr.std([1,2])
transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

img_normalized = transform_norm(img)
img_np = np.array(img_normalized)
img_normalized = np.array(img_normalized)
img_normalized = img_normalized.transpose(1, 2, 0)
plt.imshow(img_normalized.astype('uint8'))
plt.xticks([])
plt.yticks([])
plt.show()

img_nor = transform_norm(img)

# cailculate mean and std
mean, std = img_nor.mean([1, 2]), img_nor.std([1, 2])

# print mean and std
print("Mean and Std of normalized image:")
print("Mean of the image:", mean)
print("Std of the image:", std)


im  = cv2.imread('siberian-husky.jpeg')
norm_image = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

plt.imshow(norm_image)
plt.show()