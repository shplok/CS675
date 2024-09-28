import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image and create noise
img = cv2.imread('image.jpg', 1)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

height, width, _ = img.shape
noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)  # Create a random noise image
noisyImg = cv2.addWeighted(noise, 0.5, img, 0.5, 0)  # Blend it with the original image
noisyImg_rgb = cv2.cvtColor(noisyImg, cv2.COLOR_BGR2RGB)

#------------ HIGH PASS ------------#
noisyGray = cv2.cvtColor(noisyImg_rgb, cv2.COLOR_RGB2GRAY)
laplacian = cv2.Laplacian(noisyGray, cv2.CV_64F)
highFreqNoise = cv2.convertScaleAbs(laplacian)
amplifiedHighFreqNoise = cv2.addWeighted(highFreqNoise, 2.0, highFreqNoise, 0, 0)
noisyHighFreqImg = cv2.addWeighted(noisyImg, 0.8, cv2.cvtColor(amplifiedHighFreqNoise, cv2.COLOR_GRAY2BGR), 0.2, 0)
noisyHighFreqImg_rgb = cv2.cvtColor(noisyHighFreqImg, cv2.COLOR_BGR2RGB)
#-----------------------------------#

# Compute the absolute difference between the original image and the noisy image
difference = cv2.absdiff(noisyHighFreqImg_rgb, img_rgb)

# Design a filter to remove high-frequency noise based on the difference
def remove_high_freq_noise(image, difference):
    kernel = np.ones((5,5),np.float32)/25
    denoised_img = cv2.filter2D(image,-1,kernel)
    return cv2.add(denoised_img, difference)

# Apply the filter to remove high-frequency noise
denoised_img = remove_high_freq_noise(noisyHighFreqImg_rgb, difference)

# Plotting the images
plt.figure(figsize=(15, 10))

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
plt.title('High Frequency Noise')
plt.imshow(highFreqNoise, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Difference (Original vs Noisy)')
plt.imshow(difference)
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title('Denoised Image')
plt.imshow(denoised_img)
plt.axis('off')

plt.tight_layout()
plt.show()