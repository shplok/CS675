import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('image.jpg', 1)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

height, width, _ = img.shape
noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

# ---- NOISE ADDITION ----
# Blend the original image with noise
noisyImg = cv2.addWeighted(noise, 0.4, img, 0.6, 0)
noisyImg_rgb = cv2.cvtColor(noisyImg, cv2.COLOR_BGR2RGB)

# ---- HIGH PASS FILTERING ----
# Apply a slight blur before high-pass filtering
noisyGray = cv2.cvtColor(noisyImg_rgb, cv2.COLOR_RGB2GRAY)
blurred_noisyGray = cv2.GaussianBlur(noisyGray, (3, 3), 0)

# Laplacian (high-pass filter)
laplacian = cv2.Laplacian(blurred_noisyGray, cv2.CV_64F)
highFreqNoise = cv2.convertScaleAbs(laplacian)
amplifiedHighFreqNoise = cv2.addWeighted(highFreqNoise, 0.6, highFreqNoise, 0, 0)

# Subtract high-frequency noise
noisyHighFreqImg = cv2.subtract(noisyImg, cv2.cvtColor(amplifiedHighFreqNoise, cv2.COLOR_GRAY2BGR))
noisyHighFreqImg_rgb = cv2.cvtColor(noisyHighFreqImg, cv2.COLOR_BGR2RGB)

# ---- DENOISING FILTERS ----
# Box filter (Mean filter) with a larger kernel
def apply_box_filter(image):
    kernel = np.ones((7, 7), np.float32) / 49
    return cv2.filter2D(image, -1, kernel)

# Gaussian filter with a larger kernel
def apply_gaussian_filter(image):
    return cv2.GaussianBlur(image, (7, 7), 2)

# Apply denoising filters
denoised_box = apply_box_filter(noisyHighFreqImg_rgb)
denoised_gaussian = apply_gaussian_filter(noisyHighFreqImg_rgb)

# ---- PLOTTING RESULTS ----
plt.figure(figsize=(10, 5))

plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Noisy Color Image')
plt.imshow(noisyImg_rgb)
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('High Freq Noisy Image')
plt.imshow(noisyHighFreqImg_rgb)
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Denoised (Box Filter)')
plt.imshow(denoised_box)
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Denoised (Gaussian Filter)')
plt.imshow(denoised_gaussian)
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title('Difference (Original vs Denoised)')
plt.imshow(cv2.absdiff(denoised_gaussian, img_rgb))
plt.axis('off')

plt.tight_layout()
plt.show()
