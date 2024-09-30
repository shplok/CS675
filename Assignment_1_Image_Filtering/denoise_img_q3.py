import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('image.jpg', 0)  # 0 flag loads the image in grayscale

height, width = img.shape
noise = np.random.randint(0, 256, (height, width), dtype=np.uint8)

# ---- NOISE ADDITION ----
# Blend the original image with noise
noisyImg = cv2.addWeighted(noise, 0.4, img, 0.6, 0)

# ---- HIGH PASS FILTERING ----
# Apply a slight blur before high-pass filtering
blurred_noisyImg = cv2.GaussianBlur(noisyImg, (3, 3), 0)

# Laplacian (high-pass filter)
laplacian = cv2.Laplacian(blurred_noisyImg, cv2.CV_64F)
highFreqNoise = cv2.convertScaleAbs(laplacian)
amplifiedHighFreqNoise = cv2.addWeighted(highFreqNoise, 0.6, highFreqNoise, 0, 0)

# Subtract high-frequency noise
noisyHighFreqImg = cv2.subtract(noisyImg, amplifiedHighFreqNoise)

# The high frequency noise image will have a low mean and high variance 
# as it emphasizes edges and rapid changes in intensity

# ---- DENOISING FILTERS ----
# Box filter (Mean filter) with a larger kernel
def apply_box_filter(image):
    kernel = np.ones((7, 7), np.float32) / 49
    return cv2.filter2D(image, -1, kernel)

# Gaussian filter with a larger kernel
def apply_gaussian_filter(image):
    return cv2.GaussianBlur(image, (7, 7), 2)

# Apply denoising filters
denoisedBox = apply_box_filter(noisyHighFreqImg)
denoisedGaussian = apply_gaussian_filter(noisyHighFreqImg)
openCvDenoise = cv2.fastNlMeansDenoising(noisyImg, 10, 17, 21)

# All denoising methods (Box, Gaussian, and Non-local Means) should reduce the variance compared to the noisy image, 
# with the mean staying relatively close to the original image's mean

# ---- PLOTTING RESULTS ----
plt.figure(figsize=(10, 5))

plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Noisy Image')
plt.imshow(noisyImg, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('High Freq Noisy Image')
plt.imshow(noisyHighFreqImg, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Denoised (Box Filter)')
plt.imshow(denoisedBox, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Denoised (Gaussian Filter)')
plt.imshow(denoisedGaussian, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title('OpenCv Nonlocal Means Denoising')
plt.imshow(openCvDenoise, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()