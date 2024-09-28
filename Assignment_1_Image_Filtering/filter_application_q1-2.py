import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('image.jpg', 1)

#---------- BLURS ----------#
smoothBlurImg = cv2.blur(img, (3,3)) # Blur img with MxN kernel
gaussianImg = cv2.GaussianBlur(img, (3, 3), 1)
#---------------------------#

#-------- SHARPENING --------#
laplacianImg = cv2.Laplacian(img, cv2.CV_64F)
unsharpGaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
unsharpImg = cv2.addWeighted(img, 2.0, unsharpGaussian, -1.0, 0)
#----------------------------#

#---------- SOBEL ----------#
sobelImgX = cv2.Sobel(img, cv2.CV_64F, 1 ,0 ,ksize=5)
sobelImgY = cv2.Sobel(img, cv2.CV_64F, 0 ,1 ,ksize=5)
#---------------------------#

height, width, _ = img.shape
noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)  # Create a random noise image
noisyImg = cv2.addWeighted(noise, 0.5, img, 0.5, 0)  # Blend it with the original image

plt.figure(figsize=(5, 10))

# Convert images from BGR to RGB for displaying
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
smoothBlurImg_rgb = cv2.cvtColor(smoothBlurImg, cv2.COLOR_BGR2RGB)
gaussianImg_rgb = cv2.cvtColor(gaussianImg, cv2.COLOR_BGR2RGB)
unsharpImg_rgb = cv2.cvtColor(unsharpImg, cv2.COLOR_BGR2RGB)
noisyImg_rgb = cv2.cvtColor(noisyImg, cv2.COLOR_BGR2RGB)

# Create a 6x2 grid for 12 images
plt.subplot(6, 2, 1)
plt.title('Original Image')
plt.imshow(img_rgb)

# Smoothed blurred image
plt.subplot(6, 2, 2)
plt.title('Smoothed Image')
plt.imshow(smoothBlurImg_rgb)

# Original image again
plt.subplot(6, 2, 3)
plt.title('Original Image')
plt.imshow(img_rgb)

# Image with Gaussian Filter applied
plt.subplot(6, 2, 4)
plt.title('Gaussian Blurred Image')
plt.imshow(gaussianImg_rgb)

# Original image again
plt.subplot(6, 2, 5)
plt.title('Original Image')
plt.imshow(img_rgb)

# Image sharpened with Laplacian Filter
plt.subplot(6, 2, 6)
plt.title('Laplacian Sharpen')
plt.imshow(laplacianImg, cmap='gray')  # Laplacian will be in grayscale

# Original image again
plt.subplot(6, 2, 7)
plt.title('Original Image')
plt.imshow(img_rgb)

# Unsharpened image
plt.subplot(6, 2, 8)
plt.title('Unsharp Image')
plt.imshow(unsharpImg_rgb)

# Original image again
plt.subplot(6, 2, 9)
plt.title('Original Image')
plt.imshow(img_rgb)

# Image with sobel filter focused in the x-direction
plt.subplot(6, 2, 10)
plt.title('Sobel Filter X-DIR')
plt.imshow(sobelImgX, cmap='gray')  # Sobel will be in grayscale

# Original image again
plt.subplot(6, 2, 11)
plt.title('Original Image')
plt.imshow(img_rgb)

# Image with sobel filter focused in the y-direction
plt.subplot(6, 2, 12)
plt.title('Sobel Filter Y-DIR')
plt.imshow(sobelImgY, cmap='gray')  # Sobel will be in grayscale

plt.tight_layout()
plt.show()
