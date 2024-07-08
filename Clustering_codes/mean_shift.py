import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
import cv2

image = plt.imread("Images/BP6.png")
img_main = cv2.imread('Images/BP6.png')
height, width, _ = image.shape
height = int(height/2)
width = int(width/2)
image_pixels = np.reshape(img_main, (height * width, 3))#Reshape the image to a 2D array of pixels and 3 color values (RGB)

mean_s = MeanShift(bandwidth= estimate_bandwidth(image_pixels, quantile=0.1))
cluster_labels = mean_s.fit_predict(image_pixels)
cluster_centers = mean_s.cluster_centers_

print(cluster_labels[0:100]) #Resulting cluster labels

#Reshape back the image to the original dimension (width, height, 3channels)
reconstructed_image = np.reshape(cluster_centers[cluster_labels], image.shape).astype(np.uint8)

# Display the original and clustered images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Clustered Image')
plt.imshow(reconstructed_image)
plt.axis('off')
plt.show()

cv2.imshow('Original Image', img_main)
cv2.imshow('Reconstructed Image', reconstructed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()